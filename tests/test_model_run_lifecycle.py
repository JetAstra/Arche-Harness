"""research_task 模型输出展示生命周期的当前合同测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chemistry_multiagent.controllers.chemistry_multiagent_controller import (
    ResearchTaskApplication,
    _cli_result_view,
)
from chemistry_multiagent.utils.model_run_outputs import (
    ModelRunOutputWriter,
    UnsafeModelOutputPath,
)
from chemistry_multiagent.utils.model_run_retrieval import RetrievalModelOutputRecorder


def _read_json(path: Path) -> dict:
    """读取测试产生的展示 JSON。"""

    return json.loads(path.read_text(encoding="utf-8"))


def _writer(tmp_path: Path, research_task_id: str) -> ModelRunOutputWriter:
    """创建使用当前 research_task Bundle 布局的展示 writer。"""

    return ModelRunOutputWriter(
        tmp_path,
        research_task_id,
        persistence_mode="memory_on",
        scientific_question="固定科学问题",
        task_config={"entry_strategy": "standard_hypothesis_generation"},
    )


def _attempt() -> dict:
    """构造一条不含认证信息的模型调用展示收据。"""

    return {
        "request": {
            "model": "display-test-model",
            "messages": [{"role": "user", "content": "固定问题"}],
            "response_contract": {"validation_kind": "nonempty_text"},
        },
        "raw_response": {
            "id": "response-1",
            "model": "display-test-model",
            "choices": [{"message": {"content": "固定回答"}}],
        },
        "parsed_output": "固定回答",
        "validation": {"status": "valid", "failure_kind": None, "message": None},
    }


def test_writer_uses_research_task_bundle_layout(tmp_path: Path) -> None:
    """展示文件按 research_task 身份落在 outputs/research_tasks 下。"""

    writer = _writer(tmp_path, "research-task-1")

    assert writer.output_root == tmp_path / "outputs" / "research_tasks" / "research-task-1"
    assert writer.task_path.is_file()
    task = _read_json(writer.task_path)
    assert task["research_task_id"] == "research-task-1"
    assert task["persistence_mode"] == "memory_on"
    assert not (tmp_path / "outputs" / "runs").exists()


def test_stage_attempt_and_finish_update_current_audit_files(tmp_path: Path) -> None:
    """阶段、attempt 和终态写入同一份当前展示审计。"""

    writer = _writer(tmp_path, "research-task-lifecycle")
    handle = writer.start_stage(
        "retrieval",
        "summarize_retrieval",
        "01_retrieval/retrieval.json",
        input={"scientific_question": "固定科学问题"},
    )
    assert handle is not None
    assert handle.sequence == 1
    assert handle.relative_path == "internal/audit/model_calls/01_retrieval/retrieval.json"
    assert writer.append_attempt(handle, _attempt()) is True
    assert writer.finish_stage(handle, "validated", final_output={"summary": "完成"}) is True

    stage = _read_json(writer.resolve_output_path("01_retrieval/retrieval.json"))
    assert stage["status"] == "validated"
    assert stage["final_output"] == {"summary": "完成"}
    assert len(stage["attempts"]) == 1
    assert stage["attempts"][0]["validation"]["status"] == "valid"
    index = _read_json(writer.index_path)
    assert index["research_task_id"] == "research-task-lifecycle"
    assert index["calls"][0]["status"] == "validated"
    assert index["calls"][0]["attempt_count"] == 1
    assert writer.result_summary()["call_count"] == 1


def test_resume_continues_sequence_and_preserves_research_task_identity(
    tmp_path: Path,
) -> None:
    """恢复同一 research_task 时追加生命周期并继续调用序号。"""

    writer = _writer(tmp_path, "research-task-resume")
    first = writer.start_stage("retrieval", "summary", "01_retrieval/summary.json")
    assert first is not None
    assert writer.append_attempt(first, _attempt()) is True
    assert writer.finish_stage(first, "validated", final_output={"ok": True}) is True

    resumed = ModelRunOutputWriter(
        tmp_path,
        "research-task-resume",
        persistence_mode="memory_on",
        resume=True,
    )
    second = resumed.start_stage("hypothesis", "generation", "03_hypothesis/generation.json")
    assert second is not None
    assert second.sequence == 2
    assert resumed.research_task_id == writer.research_task_id
    task = _read_json(resumed.task_path)
    assert task["resume_count"] == 1
    assert [event["event"] for event in task["lifecycle"]] == [
        "started",
        "resumed",
    ]
    index = _read_json(resumed.index_path)
    assert [call["sequence"] for call in index["calls"]] == [1, 2]


def test_writer_rejects_paths_outside_current_research_task(tmp_path: Path) -> None:
    """阶段展示路径不能越出当前 research_task Bundle。"""

    writer = _writer(tmp_path, "research-task-paths")
    with pytest.raises(UnsafeModelOutputPath):
        writer.start_stage("retrieval", "unsafe", "../outside.json")


def test_retrieval_recorder_projects_valid_attempt_without_changing_scientific_result(
    tmp_path: Path,
) -> None:
    """Retrieval recorder 只写旁路展示，并保留当前阶段的有效收据。"""

    writer = _writer(tmp_path, "research-task-retrieval")
    request = {
        "scientific_question": "固定科学问题",
        "pdf_dir": "papers",
        "index_dir": "index",
        "search_papers": False,
    }
    recorder = RetrievalModelOutputRecorder(writer, request, "retrieval-operation-1")
    model_request = {
        "model": "retrieval-test-model",
        "messages": [{"role": "user", "content": "固定问题"}],
        "response_contract": {"validation_kind": "nonempty_text"},
    }
    recorder.start_model_attempt(model_request)
    recorder(
        {
            "validation_status": "valid",
            "failure_kind": None,
            "request": model_request,
            "response": {
                "provider": "fake",
                "model": "retrieval-test-model",
                "choices": [{"message": {"content": "固定检索回答"}}],
            },
        }
    )
    recorder.finish_retrieval(
        {
            "literature_review": "固定综述",
            "chemistry_context": {"charge": 0, "multiplicity": 1},
        }
    )

    stage = _read_json(writer.resolve_output_path("01_retrieval/summarize_retrieval.json"))
    assert stage["status"] == "validated"
    assert stage["attempts"][0]["validation"]["status"] == "valid"
    assert writer.result_summary()["call_count"] == 1


def test_retrieval_recorder_marks_provider_exception_as_safe_failure(
    tmp_path: Path,
) -> None:
    """Retrieval provider 异常只在展示收据中形成固定失败类别。"""

    writer = _writer(tmp_path, "research-task-retrieval-failed")
    recorder = RetrievalModelOutputRecorder(
        writer,
        {
            "scientific_question": "固定科学问题",
            "pdf_dir": "papers",
            "index_dir": "index",
            "search_papers": False,
        },
        "retrieval-operation-failed",
    )
    recorder.start_model_attempt(
        {"model": "retrieval-test-model", "messages": [{"role": "user", "content": "固定问题"}]}
    )
    recorder.finish_retrieval({"error": True})

    stage = _read_json(writer.resolve_output_path("01_retrieval/summarize_retrieval.json"))
    assert stage["status"] == "failed"
    assert stage["attempts"][0]["validation"] == {
        "status": "transport_failed",
        "failure_kind": "transport",
        "message": "模型服务请求失败",
    }
    assert stage["attempts"][0]["raw_response"] is None
    assert writer.result_summary()["failed_call_count"] == 1


def test_application_keeps_scientific_status_when_display_write_degrades(
    tmp_path: Path,
) -> None:
    """ResearchTaskApplication 附加展示摘要时不改变科学终态。"""

    application = object.__new__(ResearchTaskApplication)
    writer = _writer(tmp_path, "research-task-application")
    result = application._attach_model_output_summary(
        {
            "status": "succeeded",
            "research_task_id": "research-task-application",
        },
        writer,
    )
    assert result["status"] == "succeeded"
    assert result["model_outputs"]["status"] == "available"
    assert _read_json(writer.task_path)["status"] == "succeeded"

    degraded_writer = _writer(tmp_path, "research-task-application-degraded")

    def fail_atomic_write(_path: Path, _payload: object) -> None:
        """模拟展示目录不可写。"""

        raise OSError("simulated display write failure")

    degraded_writer._atomic_write_json = fail_atomic_write
    degraded_result = application._attach_model_output_summary(
        {
            "status": "failed",
            "research_task_id": "research-task-application-degraded",
        },
        degraded_writer,
    )
    assert degraded_result["status"] == "failed"
    assert degraded_result["model_outputs"]["status"] == "degraded"


def test_cli_result_view_exposes_formal_action_identity() -> None:
    """CLI 摘要只暴露当前 research_task/module invocation 字段。"""

    view = _cli_result_view(
        {
            "status": "waiting_investigation_job",
            "research_task_id": "research-task-1",
            "module_invocation_id": "module-invocation-1",
            "module_input_digest": "a" * 64,
            "input_scientific_record_ids": ["record-1"],
        }
    )
    assert view["research_task_id"] == "research-task-1"
    assert view["module_invocation_id"] == "module-invocation-1"
    assert view["module_input_digest"] == "a" * 64
    assert "task_id" not in view
    assert "module_run_id" not in view
