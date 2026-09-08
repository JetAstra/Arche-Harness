"""Run Bundle 存储、索引和恢复定位的快速 smoke test。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chemistry_multiagent.controllers.task_lock import TaskRunnerFileLock
from chemistry_multiagent.utils.llm_trace import _trace_dir
from chemistry_multiagent.utils.model_run_outputs import ModelRunOutputWriter
from chemistry_multiagent.utils.run_bundle import (
    RESEARCH_TASK_STAGE_NAMES,
    RunBundle,
    locate_research_task_bundle,
)


def _read_json(path: Path) -> dict:
    """读取测试 JSON。"""

    return json.loads(path.read_text(encoding="utf-8"))


def test_run_bundle_creates_updates_and_resumes_same_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """正式 research_task Bundle 的阶段摘要、状态更新和只读定位必须复用同一目录。"""

    bundle = RunBundle(
        tmp_path,
        "task-smoke",
        scientific_question="验证统一输出",
        task_config={"num_queries": 1},
    )
    state_database = bundle.root / "internal/state/arche_memory.sqlite3"
    state_database.touch()
    bundle.register_state_database(state_database)
    bundle.write_stage_summary(
        "literature_retrieval", {"status": "completed", "paper_count": 1}
    )
    bundle.update_status(
        "waiting_investigation_job",
        recovery={
            "available": True,
            "waiting_investigation_jobs": [
                {
                    "task_id": "task-smoke",
                    "batch_id": "batch-1",
                    "combination_id": "combination-1",
                    "job_id": "job-1",
                    "operation_id": "operation-1",
                    "submission_key": "submission-1",
                    "status": "waiting_investigation_job",
                }
            ],
            "pauses": [],
        },
    )

    resumed = RunBundle.open_existing_research_task(tmp_path, "task-smoke")
    assert resumed.root == bundle.root
    resumed_status = _read_json(bundle.root / "status.json")
    # 只读定位不追加新的恢复事件。
    assert resumed_status["resume_count"] == 0
    assert resumed_status["status"] == "waiting_investigation_job"
    located = locate_research_task_bundle(tmp_path, "task-smoke")
    assert located.root == bundle.root
    assert _read_json(bundle.root / "status.json") == resumed_status
    assert (
        _read_json(bundle.root / "summary.json")["stages"]["literature_retrieval"][
            "status"
        ]
        == "completed"
    )
    assert all(
        (bundle.root / "stages" / f"{stage}.json").is_file()
        for stage in RESEARCH_TASK_STAGE_NAMES
    )
    assert (tmp_path / "outputs" / "README.md").is_file()
    assert (tmp_path / "outputs" / "research_tasks" / "README.md").is_file()

    bundle.update_stage_status(
        stage="hypothesis_preparation",
        status="completed",
        summary={"status": "completed"},
    )
    assert _read_json(bundle.status_path)["recovery"]["waiting_investigation_jobs"]
    assert _read_json(bundle.summary_path)["recovery"]["waiting_investigation_jobs"]

    with pytest.raises(ValueError, match=r"waiting_investigation_jobs\[0\]"):
        bundle.update_status(
            "waiting_investigation_job",
            recovery={
                "available": True,
                "waiting_investigation_jobs": [
                    {
                        "task_id": "task-smoke",
                        "batch_id": "batch-1",
                        "combination_id": "combination-1",
                        "operation_id": "operation-1",
                        "job_id": "job-1",
                        "submission_key": None,
                        "status": "waiting_investigation_job",
                    }
                ],
                "pauses": [],
            },
        )
    bundle.update_status("running")
    assert _read_json(bundle.status_path)["recovery"] == {
        "available": False,
        "waiting_investigation_jobs": [],
        "pauses": [],
    }
    with pytest.raises(ValueError, match="运行状态非法"):
        bundle.update_status(None)  # type: ignore[arg-type]
    bundle.write_stage_summary(
        "hypothesis_preparation", {"status": "failed", "reason": "模型失败"}
    )
    assert (
        _read_json(bundle.root / "stages/hypothesis_preparation.json")["status"]
        == "failed"
    )
    hypothesis_summary = _read_json(bundle.summary_path)["stages"][
        "hypothesis_preparation"
    ]
    assert hypothesis_summary["status"] == "failed"
    assert hypothesis_summary["summary"]["reason"] == "模型失败"
    with pytest.raises(ValueError, match="restart_allowed"):
        bundle.update_status(
            "paused",
            recovery={
                "available": True,
                "waiting_executions": [],
                "pauses": [
                    {
                        "pause_id": "pause-1",
                        "operation": "execute_step",
                        "reason_code": "technical_failure",
                        "restart_allowed": None,
                    }
                ],
            },
        )

    corrupted = RunBundle(
        tmp_path,
        "task-corrupted",
    )
    corrupted_run = _read_json(corrupted.run_path)
    corrupted_run["research_task_id"] = "different-task"
    corrupted.run_path.write_text(json.dumps(corrupted_run), encoding="utf-8")
    with pytest.raises(ValueError, match="research_task_id 不一致"):
        RunBundle.open_existing_research_task(tmp_path, "task-corrupted")

    corrupted_stage = RunBundle(
        tmp_path,
        "task-corrupted-stage",
    )
    stage_path = corrupted_stage.root / "stages" / "literature_retrieval.json"
    stage_document = _read_json(stage_path)
    stage_document["stage"] = "planning"
    stage_path.write_text(json.dumps(stage_document), encoding="utf-8")
    with pytest.raises(ValueError, match="阶段身份不一致"):
        RunBundle.open_existing_research_task(tmp_path, "task-corrupted-stage")

    symlink_bundle = RunBundle(
        tmp_path,
        "task-symlink",
    )
    locks_directory = symlink_bundle.root / ".locks"
    locks_directory.rmdir()
    locks_directory.symlink_to(tmp_path / "external-locks", target_is_directory=True)
    with pytest.raises(ValueError, match="固定目录非法"):
        RunBundle.open_existing_research_task(tmp_path, "task-symlink")

    lock_root = tmp_path / "task-lock-symlink"
    lock_parent = lock_root / ".locks"
    lock_parent.mkdir(parents=True)
    lock_path = lock_parent / "task_runner.lock"
    lock_path.symlink_to(tmp_path / "external-task-runner.lock")
    with pytest.raises(ValueError, match="任务锁文件不得"):
        TaskRunnerFileLock(lock_path).acquire()

    resumed.update_status("paused", reason_code="manual_pause")
    paused_resume = RunBundle.open_existing_research_task(tmp_path, "task-smoke")
    assert _read_json(paused_resume.status_path)["status"] == "paused"

    second_bundle = RunBundle(
        tmp_path,
        "task-other",
    )
    second_state_database = second_bundle.root / "internal/state/arche_memory.sqlite3"
    second_state_database.touch()
    second_bundle.register_state_database(second_state_database)
    assert (
        second_bundle.locate_research_task_state_database()
        != bundle.locate_research_task_state_database()
    )

    writer = ModelRunOutputWriter(
        tmp_path,
        "task-smoke",
        bundle_root=bundle.root,
        persistence_mode="memory_on",
        resume=False,
    )
    assert writer.output_root == bundle.root
    with pytest.raises(ValueError, match="bundle_root 必须等于"):
        ModelRunOutputWriter(
            tmp_path,
            "task-outside",
            bundle_root=tmp_path / "outside",
        )
    assert not (tmp_path / "outside").exists()
    handle = writer.start_stage(
        "literature_retrieval",
        "summarize_retrieval",
        "01_retrieval/summarize.json",
    )
    assert handle is not None
    assert writer.append_attempt(handle, {})
    assert writer.finish_stage(handle, "validated", final_output={"paper_count": 1})
    assert (
        bundle.root / "internal/audit/model_calls/01_retrieval/summarize.json"
    ).is_file()
    retrieval_stage = _read_json(
        bundle.root / "stages/literature_retrieval.json"
    )
    assert retrieval_stage["status"] == "completed"
    assert retrieval_stage["entries"]
    assert _read_json(bundle.root / "summary.json")["stages"]["literature_retrieval"][
        "entry_count"
    ] == len(retrieval_stage["entries"])
    model_call = _read_json(
        bundle.root / "internal/audit/model_calls/01_retrieval/summarize.json"
    )
    bundle.write_stage_summary(
        "literature_retrieval",
        {"status": "completed", "paper_count": 2},
        source_file="internal/audit/model_calls/01_retrieval/summarize.json",
        call_identity=model_call["call_identity"],
    )
    assert len(
        _read_json(bundle.root / "stages/literature_retrieval.json")["entries"]
    ) == len(
        retrieval_stage["entries"]
    )
    assert writer.update_run_bundle(
        {"status": "retrieval_incomplete", "reason_code": "paper_search_incomplete"}
    )
    assert _read_json(bundle.root / "status.json")["status"] == "retrieval_incomplete"
    assert writer.update_run_bundle(
        {
            "status": "waiting_investigation_job",
            "recovery": {
                "available": True,
                "waiting_investigation_jobs": [
                    {
                        "task_id": "task-smoke",
                        "batch_id": "batch-1",
                        "combination_id": "combination-2",
                        "job_id": "job-2",
                        "operation_id": "operation-2",
                        "submission_key": "submission-2",
                        "status": "waiting_investigation_job",
                    }
                ],
                "pauses": [],
            },
        }
    )
    waiting_status = _read_json(bundle.root / "status.json")
    assert waiting_status["status"] == "waiting_investigation_job"
    assert (
        waiting_status["recovery"]["waiting_investigation_jobs"][0]["job_id"]
        == "job-2"
    )
    assert ModelRunOutputWriter._canonical_bundle_stage(
        "query", "generate_queries"
    ) == "hypothesis_preparation"
    assert ModelRunOutputWriter._canonical_bundle_stage(
        "retrieval", "summarize_retrieval"
    ) == "literature_retrieval"
    bundle.update_stage_status(
        stage="task_review_and_result",
        status="completed",
        summary={"status": "succeeded", "output_files": {"complete_results": []}},
    )
    final_summary = _read_json(bundle.root / "summary.json")
    assert final_summary["stages"]["task_review_and_result"]["status"] == "completed"
    assert final_summary["final_result"]["status"] == "succeeded"
    assert writer.update_run_bundle({"status": "succeeded"})
    assert _read_json(bundle.root / "status.json")["recovery"] == {
        "available": False,
        "waiting_investigation_jobs": [],
        "pauses": [],
    }
    assert _read_json(bundle.root / "summary.json")["recovery"] == {
        "available": False,
        "waiting_investigation_jobs": [],
        "pauses": [],
    }
    assert (bundle.root / "internal/audit/index.json").is_file()
    assert not (tmp_path / "outputs" / "model_runs").exists()
    assert not (tmp_path / "outputs" / "llm_traces").exists()
    assert not (tmp_path / ".runtime/tasks/outputs").exists()

    monkeypatch.setenv("ARCHE_LLM_TRACE_DIR", str(tmp_path / "external-traces"))
    monkeypatch.setenv("ARCHE_RUN_BUNDLE_ROOT", str(bundle.root))
    assert _trace_dir() == bundle.root / "internal/audit/llm_traces"
    monkeypatch.delenv("ARCHE_RUN_BUNDLE_ROOT")
    assert _trace_dir() == tmp_path / "external-traces"
    monkeypatch.delenv("ARCHE_LLM_TRACE_DIR")
    assert _trace_dir() == Path("local_data/runtime/diagnostics/llm_traces")
