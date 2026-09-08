"""MiniChem workspace text 大产物读取的公开 MCP 合同测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from minichem_mcp_tools.execution_models import (
    ArtifactDeclarationRequest,
    StagedInput,
    WorkspaceTextReadRequest,
)
from minichem_mcp_tools.open_execution import read_workspace_text, write_workspace_text
from minichem_mcp_tools.open_execution import (
    _declare_scientific_artifact_once,
    _stage_inputs,
)
from minichem_mcp_tools.execution_models import WorkspaceTextWriteRequest
from minichem_mcp_tools.open_tools import TOOL_DESCRIPTIONS


def _write_workspace_file(root: Path, relative_path: str, content: str) -> None:
    """在隔离 workspace 中写入测试文件。"""

    target = root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def test_workspace_text_schema_exposes_paging_and_json_pointer_fields() -> None:
    """公开 Pydantic schema 必须暴露续读和 JSON Pointer 请求字段。"""

    schema = WorkspaceTextReadRequest.model_json_schema()
    properties = schema["properties"]

    assert properties["offset"]["minimum"] == 0
    assert properties["offset"]["default"] == 0
    assert "maximum" not in properties["offset"]
    assert "json_pointer" in properties
    assert properties["json_pointer"]["anyOf"]
    description = TOOL_DESCRIPTIONS["read_workspace_text"]
    assert "offset" in description
    assert "json_pointer" in description
    assert "next_offset" in description


def test_workspace_text_read_reports_character_pages_and_eof(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """服务端按字符偏移分页，并在最后一页返回 null next_offset。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    content = "甲乙丙丁戊"
    _write_workspace_file(tmp_path, "outputs/large.txt", content)

    first = read_workspace_text(
        WorkspaceTextReadRequest(path="outputs/large.txt", max_chars=2)
    )
    second = read_workspace_text(
        WorkspaceTextReadRequest(
            path="outputs/large.txt", max_chars=2, offset=first["next_offset"]
        )
    )
    final = read_workspace_text(
        WorkspaceTextReadRequest(
            path="outputs/large.txt", max_chars=2, offset=second["next_offset"]
        )
    )

    assert first["content"] + second["content"] + final["content"] == content
    assert first["offset"] == 0
    assert first["returned_chars"] == 2
    assert first["next_offset"] == 2
    assert first["total_chars"] == len(content)
    assert first["truncated"] is True
    assert final["offset"] == 4
    assert final["returned_chars"] == 1
    assert final["next_offset"] is None
    assert final["truncated"] is False

    eof = read_workspace_text(
        WorkspaceTextReadRequest(path="outputs/large.txt", max_chars=2, offset=len(content))
    )
    assert eof["content"] == ""
    assert eof["returned_chars"] == 0
    assert eof["next_offset"] is None
    assert eof["truncated"] is False

    far_eof = read_workspace_text(
        WorkspaceTextReadRequest(
            path="outputs/large.txt", max_chars=2, offset=10_000_001
        )
    )
    assert far_eof["offset"] == 10_000_001
    assert far_eof["content"] == ""
    assert far_eof["next_offset"] is None


def test_workspace_text_read_supports_root_nested_and_escaped_json_pointers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """JSON Pointer 必须支持根、对象、数组以及 RFC 6901 转义键。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    value = {
        "z": 1,
        "a/b": {"~key": ["zero", {"message": "中/\\\""}]},
    }
    _write_workspace_file(
        tmp_path,
        "outputs/result.json",
        json.dumps(value, ensure_ascii=False, indent=2),
    )

    root = read_workspace_text(
        WorkspaceTextReadRequest(path="outputs/result.json", json_pointer="")
    )
    nested = read_workspace_text(
        WorkspaceTextReadRequest(
            path="outputs/result.json",
            json_pointer="/a~1b/~0key/1",
            max_chars=100,
        )
    )
    array_item = read_workspace_text(
        WorkspaceTextReadRequest(
            path="outputs/result.json", json_pointer="/a~1b/~0key/0"
        )
    )

    assert root["json_pointer"] == ""
    assert root["content"] == json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    assert nested["json_pointer"] == "/a~1b/~0key/1"
    assert nested["content"] == json.dumps(
        value["a/b"]["~key"][1],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert array_item["content"] == '"zero"'
    assert nested["total_chars"] == len(nested["content"])
    assert nested["next_offset"] is None


@pytest.mark.parametrize(
    "pointer",
    ["/missing", "/a~2b", "/a~1b/~0key/01", "/a~1b/~0key/-", "/a~1b/~0key/0/message"],
)
def test_workspace_text_read_rejects_invalid_json_pointer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pointer: str
) -> None:
    """不存在、非法转义、非法数组下标和标量后缀都必须明确失败。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    _write_workspace_file(
        tmp_path,
        "outputs/result.json",
        json.dumps({"a/b": {"~key": ["zero"]}}, ensure_ascii=False),
    )

    with pytest.raises(ValueError):
        read_workspace_text(
            WorkspaceTextReadRequest(path="outputs/result.json", json_pointer=pointer)
        )


def test_workspace_text_read_rejects_invalid_json_for_pointer_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """指定 JSON Pointer 时，无效 UTF-8 JSON 不能退化成普通文本读取。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    _write_workspace_file(tmp_path, "outputs/bad.json", "{not-json")

    with pytest.raises(ValueError):
        read_workspace_text(
            WorkspaceTextReadRequest(path="outputs/bad.json", json_pointer="")
        )


@pytest.mark.parametrize(
    "relative_path",
    [
        "_toolbox_catalog.json",
        "_tool_trace.jsonl",
        "_tool_artifacts/operations/receipt.json",
        "_tool_results/.sequence-0001",
        ".runtime/state/checkpoint.json",
        "internal/trace/receipt.json",
        "outputs/execution_jobs/job_123/status.json",
        "outputs/.resource_budget",
        "tool_logs/server.log",
        "logs/internal.log",
        "config.local.env",
        ".env",
        ".git/config",
        ".ssh/id_ed25519",
    ],
)
def test_workspace_text_read_rejects_framework_internal_paths_without_disclosure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, relative_path: str
) -> None:
    """文本读取不得暴露框架控制状态，错误不得回显绝对路径。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    secret = "INTERNAL_PATH_CONTENT_SENTINEL"
    _write_workspace_file(tmp_path, relative_path, secret)

    with pytest.raises(ValueError) as error:
        read_workspace_text(WorkspaceTextReadRequest(path=relative_path))

    message = str(error.value)
    assert str(tmp_path) not in message
    assert secret not in message


@pytest.mark.parametrize(
    "relative_path",
    [
        "_toolbox_catalog.json",
        "_tool_trace.jsonl",
        "_tool_artifacts/operations/receipt.json",
        "_tool_results/.sequence-0001",
        ".runtime/state/checkpoint.json",
        "internal/trace/receipt.json",
        "outputs/execution_jobs/job_123/status.json",
        "outputs/.resource_budget",
        "tool_logs/server.log",
        "logs/internal.log",
        "config.local.env",
        ".env",
        ".git/config",
        ".ssh/id_ed25519",
    ],
)
def test_workspace_text_write_rejects_framework_internal_paths_without_overwrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, relative_path: str
) -> None:
    """文本写入不得覆盖框架控制状态，错误不得回显绝对路径。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    target = tmp_path / relative_path
    _write_workspace_file(tmp_path, relative_path, "ORIGINAL_INTERNAL_CONTENT")

    with pytest.raises(ValueError) as error:
        write_workspace_text(
            WorkspaceTextWriteRequest(
                path=relative_path,
                content="REPLACEMENT_INTERNAL_CONTENT",
                overwrite=True,
            )
        )

    message = str(error.value)
    assert str(tmp_path) not in message
    assert "REPLACEMENT_INTERNAL_CONTENT" not in message
    assert target.read_text(encoding="utf-8") == "ORIGINAL_INTERNAL_CONTENT"


@pytest.mark.parametrize(
    "relative_path",
    [
        "outputs/action/result.xyz",
        "outputs/action/reaction_trace.json",
        "outputs/action/checkpoint_geometry.xyz",
        "code/input.gjf",
        "report/final.md",
    ],
)
def test_workspace_text_keeps_explicit_user_files_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, relative_path: str
) -> None:
    """普通输出、代码输入和报告文件仍可分页读取并按显式覆盖写入。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    _write_workspace_file(tmp_path, relative_path, "old")

    written = write_workspace_text(
        WorkspaceTextWriteRequest(path=relative_path, content="new", overwrite=True)
    )
    read = read_workspace_text(WorkspaceTextReadRequest(path=relative_path))

    assert written["path"] == relative_path
    assert read["content"] == "new"


def test_workspace_text_reads_execution_job_result_but_not_job_control_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """执行作业产生的科学结果可读，状态文件仍保持内部隔离。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    result_path = "outputs/execution_jobs/job_123/outputs/result.xyz"
    _write_workspace_file(tmp_path, result_path, "RESULT_CONTENT")
    _write_workspace_file(
        tmp_path,
        "outputs/execution_jobs/job_123/status.json",
        "CONTROL_CONTENT",
    )

    result = read_workspace_text(WorkspaceTextReadRequest(path=result_path))

    assert result["content"] == "RESULT_CONTENT"
    with pytest.raises(ValueError):
        read_workspace_text(
            WorkspaceTextReadRequest(
                path="outputs/execution_jobs/job_123/status.json"
            )
        )
    with pytest.raises(ValueError):
        write_workspace_text(
            WorkspaceTextWriteRequest(
                path=result_path, content="REPLACEMENT_RESULT", overwrite=True
            )
        )


def test_workspace_text_keeps_action_artifact_object_readable_but_immutable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Action 的完整 Artifact 对象可按引用读取，但不能由文本写工具覆盖。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    artifact_path = "_tool_artifacts/objects/art_0123456789abcdef0123456789abcdef.json"
    _write_workspace_file(tmp_path, artifact_path, '{"value":"artifact"}')

    result = read_workspace_text(WorkspaceTextReadRequest(path=artifact_path))

    assert result["content"] == '{"value":"artifact"}'
    with pytest.raises(ValueError):
        write_workspace_text(
            WorkspaceTextWriteRequest(
                path=artifact_path, content="replacement", overwrite=True
            )
        )


def test_workspace_text_rejects_non_artifact_json_inside_object_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Artifact 对象白名单只接受服务生成的不可预测 Artifact 文件名。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    internal_path = "_tool_artifacts/objects/config.json"
    _write_workspace_file(tmp_path, internal_path, '{"internal":true}')

    with pytest.raises(ValueError, match="reserved"):
        read_workspace_text(WorkspaceTextReadRequest(path=internal_path))


def test_execution_job_result_can_be_staged_and_declared_as_scientific_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """执行作业的科学结果仍可进入后续作业或声明为 Artifact。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    result_path = "outputs/execution_jobs/job_123/outputs/result.xyz"
    _write_workspace_file(tmp_path, result_path, "RESULT_CONTENT")
    job_directory = tmp_path / "outputs" / "execution_jobs" / "job_456"
    job_directory.mkdir(parents=True)

    records = _stage_inputs(
        job_directory,
        [StagedInput(source_path=result_path, target_path="inputs/result.xyz")],
    )
    artifact = _declare_scientific_artifact_once(
        ArtifactDeclarationRequest(
            path=result_path,
            semantic_type="AtomicStructure",
            media_type="chemical/x-xyz",
            producer_layer="native_software",
            producer_id="test",
        )
    )

    assert records[0]["source_path"] == result_path
    assert (job_directory / "inputs" / "result.xyz").read_text(encoding="utf-8") == (
        "RESULT_CONTENT"
    )
    assert artifact["artifact"]["path"] == result_path


def test_workspace_text_rejects_symlink_parent_before_internal_path_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """符号链接父目录不能把文本读写绕到工作区其他位置。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    target_root = tmp_path / "outside"
    target_root.mkdir()
    _write_workspace_file(target_root, "_toolbox_catalog.json", "LINKED_INTERNAL_CONTENT")
    (tmp_path / "outputs").symlink_to(target_root, target_is_directory=True)

    with pytest.raises(ValueError):
        read_workspace_text(
            WorkspaceTextReadRequest(path="outputs/_toolbox_catalog.json")
        )
    with pytest.raises(ValueError):
        write_workspace_text(
            WorkspaceTextWriteRequest(
                path="outputs/_toolbox_catalog.json", content="replacement", overwrite=True
            )
        )
