"""ModelRunOutputWriter 的独立展示存储协议测试。"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from chemistry_multiagent.utils.model_run_outputs import (
    ModelRunOutputWriter,
    ModelStageHandle,
    UnsafeModelOutputPath,
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _writer(
    tmp_path: Path, research_task_id: str = "diagnostic-test"
) -> ModelRunOutputWriter:
    return ModelRunOutputWriter(
        tmp_path,
        research_task_id,
        persistence_mode="memory_off",
        scientific_question="验证展示文件中的中文和生命周期。",
        task_config={"num_queries": 1},
    )


def _artifact(relative_path: str, *, size: int = 0) -> dict[str, object]:
    """构造不触碰文件系统的合法 ArtifactRef。"""

    return {
        "relative_path": relative_path,
        "sha256": "a" * 64,
        "media_type": "application/json",
        "size": size,
    }


@pytest.mark.parametrize(
    ("phase", "stage", "expected"),
    [
        ("retrieval", "retrieval", "literature_retrieval"),
        (
            "research-query-formulation",
            "research-query-formulation",
            "hypothesis_preparation",
        ),
        (
            "hypothesis-generation",
            "hypothesis-generation",
            "hypothesis_preparation",
        ),
        (
            "hypothesis-selection",
            "hypothesis-selection",
            "hypothesis_preparation",
        ),
        (
            "hypothesis-confirmation",
            "hypothesis-confirmation",
            "hypothesis_preparation",
        ),
        (
            "hypothesis-investigation",
            "hypothesis-investigation",
            "investigation_and_experiment",
        ),
        ("task-review", "task-review", "task_review_and_result"),
    ],
)
def test_formal_module_names_map_to_formal_bundle_stages(
    phase: str,
    stage: str,
    expected: str,
) -> None:
    """正式模块名必须投影到四个正式 Bundle 阶段。"""

    assert ModelRunOutputWriter._canonical_bundle_stage(phase, stage) == expected


def test_writer_rejects_historical_waiting_status_without_projecting_recovery(
    tmp_path: Path,
) -> None:
    """展示 writer 不解释历史等待状态，也不从检查点生成恢复事实。"""

    writer = _writer(tmp_path, "diagnostic-formal-status")

    with pytest.raises(ValueError, match="research_task 状态非法"):
        writer.update_run_bundle(
            {
                "status": "waiting_execution",
                "runtime_checkpoint": {
                    "operation_id": "legacy-operation",
                    "checkpoint_id": "legacy-checkpoint",
                },
            }
        )

    status = _read_json(writer.bundle_root / "status.json")
    assert status["status"] == "running"
    assert "waiting_executions" not in status["recovery"]
    assert status["recovery"]["waiting_investigation_jobs"] == []


def test_stage_files_use_readable_json_and_do_not_create_empty_phase_dirs(tmp_path):
    writer = _writer(tmp_path)
    root = tmp_path / "outputs" / "research_tasks" / "diagnostic-test"

    assert sorted(path.name for path in root.iterdir()) == [
        ".bundle.lock",
        ".locks",
        "README.md",
        "artifacts",
        "internal",
        "run.json",
        "stages",
        "status.json",
        "summary.json",
    ]
    task = _read_json(root / "internal/audit/task.json")
    index = _read_json(root / "internal/audit/index.json")
    assert list(task) == [
        "schema_version",
        "research_task_id",
        "persistence_mode",
        "scientific_question",
        "status",
        "started_at",
        "finished_at",
        "resume_count",
        "task_config",
        "model_configuration",
        "tool_context",
        "source_locations",
        "model_output_status",
        "model_output_root",
        "lifecycle",
    ]
    assert list(index) == [
        "schema_version",
        "research_task_id",
        "updated_at",
        "calls",
    ]
    assert not (root / "internal/audit/model_calls/02_query").exists()

    handle = writer.start_stage(
        "query",
        "query_generation",
        "02_query/batch_001/generate_queries.json",
        input={"scientific_question": "苯甲醛"},
    )
    assert handle is not None
    writer.append_attempt(
        handle,
        {
            "provider": "test-provider",
            "model": "test-model",
            "request": {
                "catalog_key": "hypothesis_generation.query",
                "model": "test-model",
                "messages": [{"role": "user", "content": "苯甲醛"}],
                "headers": {"Authorization": "must-not-be-saved"},
                "env": {"API_KEY": "must-not-be-saved"},
            },
            "raw_response": {
                "id": "provider-id-must-not-be-saved",
                "created": 123,
                "provider": "test-provider",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "回答"},
                    }
                ],
                "usage": {"prompt_tokens": 3, "unexpected": "drop"},
                "headers": {"Authorization": "must-not-be-saved"},
            },
            "parsed_output": {"queries": ["问题一"], "api_key": "drop"},
            "validation": {"status": "valid", "message": None},
        },
    )
    assert writer.finish_stage(handle, "validated", final_output={"queries": ["问题一"]})

    stage_path = writer.resolve_output_path(
        "02_query/batch_001/generate_queries.json"
    )
    raw_text = stage_path.read_text(encoding="utf-8")
    saved = json.loads(raw_text)
    assert raw_text.endswith("\n")
    assert "苯甲醛" in raw_text
    assert "\\u82ef" not in raw_text
    assert saved["status"] == "validated"
    attempt = saved["attempts"][0]
    assert attempt["attempt_index"] == 0
    assert attempt["provider_request_id"] is None
    assert attempt["request"] == {
        "catalog_key": "hypothesis_generation.query",
        "model": "test-model",
        "messages": [{"role": "user", "content": "苯甲醛"}],
    }
    assert attempt["raw_response"] == {
        "id": "provider-id-must-not-be-saved",
        "created": 123,
        "provider": "test-provider",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "回答"},
            }
        ],
        "usage": {"prompt_tokens": 3},
    }
    assert not list(root.rglob("*must-not-be-saved*"))
    assert sorted(path.name for path in (root / "artifacts").iterdir()) == [
        "README.md"
    ]


def test_scientific_payload_keeps_literal_sensitive_named_fields(tmp_path):
    writer = _writer(tmp_path, "diagnostic-science-fields")
    handle = writer.start_stage(
        "query",
        "query_generation",
        "02_query/query.json",
        input={"api_key": "合法科学字段", "token": "化学 token", "secret": "正文"},
    )
    assert handle is not None
    assert writer.append_attempt(
        handle,
        {
            "parsed_output": {"api_key": "保留", "token": "保留"},
            "raw_response": {
                "provider": {"Authorization": "丢弃"},
                "access_token": "丢弃",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": {"key": "正文"},
                            "reasoning_content": "推理正文",
                            "headers": {"cookie": "丢弃"},
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '{"token":"正文"}',
                                    },
                                    "access_token": "丢弃",
                                }
                            ],
                        }
                    }
                ],
            },
        },
    )
    assert writer.finish_stage(
        handle,
        "validated",
        final_output={"secret": "最终科学正文", "token": "最终 token"},
    )
    saved = _read_json(writer.resolve_output_path("02_query/query.json"))
    assert saved["input"]["api_key"] == "合法科学字段"
    assert saved["attempts"][0]["parsed_output"]["api_key"] == "保留"
    response = saved["attempts"][0]["raw_response"]
    assert "access_token" not in response
    assert response["choices"][0]["message"]["reasoning_content"] == "推理正文"
    assert response["choices"][0]["message"]["content"] == {"key": "正文"}
    assert response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == (
        '{"token":"正文"}'
    )
    assert saved["final_output"]["secret"] == "最终科学正文"


def test_audit_preserves_unset_compiled_output_budget_as_null(tmp_path):
    """展示审计保留 compiled request 的 null 预算事实，不替换成默认上限。"""

    writer = _writer(tmp_path, "diagnostic-null-budget")
    handle = writer.start_stage("query", "query_generation", "02_query/query.json")
    assert handle is not None
    assert writer.append_attempt(
        handle,
        {
            "request": {
                "model": "configured-model",
                "max_tokens": None,
                "reasoning_effort": "max",
                "provider_output_mode": "json_object",
                "provider_response_format": None,
                "messages": [{"role": "user", "content": "测试"}],
            },
            "raw_response": {"choices": []},
        },
    )
    assert writer.finish_stage(handle, "validated")

    saved = _read_json(writer.output_root / handle.relative_path)
    assert saved["attempts"][0]["request"]["max_tokens"] is None
    assert saved["attempts"][0]["request"]["reasoning_effort"] == "max"
    assert saved["attempts"][0]["request"]["provider_output_mode"] == "json_object"
    assert saved["attempts"][0]["request"]["provider_response_format"] is None


def test_malformed_response_containers_are_emptied_without_copying_headers(tmp_path):
    writer = _writer(tmp_path, "diagnostic-malformed-response")
    handle = writer.start_stage("query", "query_generation", "02_query/query.json")
    assert handle is not None
    assert writer.append_attempt(
        handle,
        {
            "request": {
                "messages": [
                    {
                        "role": "user",
                        "content": "保留",
                        "tool_calls": {"Authorization": "must-not-be-saved"},
                    }
                ]
            },
            "raw_response": {
                "choices": {"Authorization": "must-not-be-saved"},
                "usage": ["Bearer must-not-be-saved"],
                "headers": {"Cookie": "must-not-be-saved"},
            }
        },
    )
    assert writer.append_attempt(
        handle,
        {
            "raw_response": {
                "choices": [
                    {
                        "message": {
                            "Authorization": "must-not-be-saved",
                            "tool_calls": {"Cookie": "must-not-be-saved"},
                        }
                    }
                ]
            }
        },
    )
    assert writer.finish_stage(handle, "failed")

    attempts = _read_json(writer.resolve_output_path("02_query/query.json"))["attempts"]
    response = attempts[0]["raw_response"]
    request = attempts[0]["request"]
    nested_response = attempts[1]["raw_response"]
    assert response["choices"] == []
    assert response["usage"] == {}
    assert "headers" not in response
    assert "must-not-be-saved" not in json.dumps(response, ensure_ascii=False)
    assert request["messages"][0]["tool_calls"] == []
    assert "must-not-be-saved" not in json.dumps(request, ensure_ascii=False)
    assert nested_response["choices"][0]["message"] == {"tool_calls": []}
    assert "must-not-be-saved" not in json.dumps(nested_response, ensure_ascii=False)


def test_concurrent_stages_receive_unique_sequences(tmp_path):
    writer = _writer(tmp_path, "diagnostic-concurrent")
    handles = []
    errors = []
    barrier = threading.Barrier(12)

    def create_stage(index: int) -> None:
        try:
            barrier.wait()
            handle = writer.start_stage(
                "hypothesis",
                "generation",
                f"03_hypothesis/generation/query_{index:03d}.json",
            )
            assert handle is not None
            handles.append(handle)
        except Exception as error:  # pragma: no cover - 失败信息由断言报告
            errors.append(error)

    threads = [threading.Thread(target=create_stage, args=(index,)) for index in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert sorted(handle.sequence for handle in handles) == list(range(1, 13))
    index = _read_json(
        tmp_path
        / "outputs/research_tasks/diagnostic-concurrent/internal/audit/index.json"
    )
    assert [call["sequence"] for call in index["calls"]] == list(range(1, 13))


def test_resume_continues_after_largest_index_sequence_and_records_lifecycle(tmp_path):
    writer = ModelRunOutputWriter(
        tmp_path,
        "task-resume",
        persistence_mode="memory_on",
    )
    first = writer.start_stage("query", "query_generation", "02_query/query.json")
    assert first is not None and first.sequence == 1

    resumed = ModelRunOutputWriter(
        tmp_path,
        "task-resume",
        persistence_mode="memory_on",
        resume=True,
    )
    second = resumed.start_stage("router", "evidence", "05_router/router.json")
    assert second is not None and second.sequence == 2

    task = _read_json(
        tmp_path / "outputs/research_tasks/task-resume/internal/audit/task.json"
    )
    assert task["resume_count"] == 1
    assert task["lifecycle"][-1]["event"] == "resumed"
    index = _read_json(
        tmp_path / "outputs/research_tasks/task-resume/internal/audit/index.json"
    )
    assert [call["sequence"] for call in index["calls"]] == [1, 2]


@pytest.mark.parametrize("kind", ["wrong_mode", "wrong_schema", "wrong_index_runtime"])
def test_resume_rejects_identity_or_schema_drift_without_rewriting_files(tmp_path, kind):
    writer = _writer(tmp_path, f"task-drift-{kind}")
    root = writer.output_root
    task_path = root / "internal/audit/task.json"
    index_path = root / "internal/audit/index.json"
    task_before = task_path.read_text(encoding="utf-8")
    index_before = index_path.read_text(encoding="utf-8")
    if kind == "wrong_mode":
        resumed = ModelRunOutputWriter(
            tmp_path, f"task-drift-{kind}", persistence_mode="memory_on", resume=True
        )
    elif kind == "wrong_schema":
        task = _read_json(task_path)
        task["schema_version"] = 99
        task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        resumed = ModelRunOutputWriter(tmp_path, f"task-drift-{kind}", resume=True)
    else:
        index = _read_json(index_path)
        index["research_task_id"] = "other-research-task"
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        resumed = ModelRunOutputWriter(tmp_path, f"task-drift-{kind}", resume=True)
    assert resumed.result_summary()["status"] == "degraded"
    if kind == "wrong_mode":
        assert task_path.read_text(encoding="utf-8") == task_before
        assert index_path.read_text(encoding="utf-8") == index_before


def test_resume_missing_task_does_not_create_an_empty_display_directory(tmp_path):
    writer = ModelRunOutputWriter(
        tmp_path,
        "task-missing",
        persistence_mode="memory_on",
        resume=True,
    )
    root = tmp_path / "outputs" / "research_tasks" / "task-missing"

    assert not root.exists()
    assert writer.result_summary()["status"] == "degraded"


def test_resume_broken_files_does_not_rebuild_old_task(tmp_path):
    root = (
        tmp_path
        / "outputs"
        / "research_tasks"
        / "task-broken"
        / "internal"
        / "audit"
    )
    root.mkdir(parents=True)
    (root / "task.json").write_text("{broken", encoding="utf-8")
    (root / "index.json").write_text("{broken", encoding="utf-8")

    writer = ModelRunOutputWriter(tmp_path, "task-broken", resume=True)

    assert (root / "task.json").read_text(encoding="utf-8") == "{broken"
    assert (root / "index.json").read_text(encoding="utf-8") == "{broken"
    assert writer.result_summary()["status"] == "degraded"


@pytest.mark.parametrize("symlink_parent", ["outputs", "research_tasks"])
def test_resume_rejects_symlinked_storage_parent_without_touching_external_path(
    tmp_path, symlink_parent
):
    external_root = tmp_path / "external-storage"
    external_root.mkdir()
    outputs = tmp_path / "outputs"
    if symlink_parent == "outputs":
        outputs.symlink_to(external_root, target_is_directory=True)
    else:
        outputs.mkdir()
        (outputs / "research_tasks").symlink_to(
            external_root, target_is_directory=True
        )

    writer = ModelRunOutputWriter(
        tmp_path,
        "task-symlink-parent",
        persistence_mode="memory_on",
        resume=True,
    )

    assert writer.result_summary()["status"] == "degraded"
    assert not (external_root / "task.json").exists()
    assert not (external_root / "index.json").exists()


def test_terminal_stage_cannot_be_overwritten(tmp_path):
    writer = _writer(tmp_path, "diagnostic-terminal")
    handle = writer.start_stage("router", "evidence", "05_router/evidence.json")
    assert handle is not None
    assert writer.append_attempt(handle, {})
    assert writer.finish_stage(handle, "validated", final_output={"decision": "ready"})
    path = writer.resolve_output_path("05_router/evidence.json")
    before = path.read_text(encoding="utf-8")

    assert not writer.append_attempt(handle, {"raw_response": {"choices": []}})
    assert not writer.finish_stage(handle, "failed")
    assert path.read_text(encoding="utf-8") == before


def test_terminal_path_collision_marks_new_call_degraded(tmp_path):
    writer = _writer(tmp_path, "diagnostic-collision")
    handle = writer.start_stage("router", "evidence", "05_router/evidence.json")
    assert handle is not None
    assert writer.append_attempt(handle, {})
    assert writer.finish_stage(handle, "validated", final_output={"decision": "ready"})

    assert writer.start_stage("router", "evidence", "05_router/evidence.json") is None
    assert writer.result_summary()["status"] == "degraded"


def test_running_stage_identity_mismatch_is_not_reused(tmp_path):
    writer = _writer(tmp_path, "diagnostic-identity")
    handle = writer.start_stage(
        "router", "evidence", "05_router/evidence.json", operation_id="op-1"
    )
    assert handle is not None
    before = writer.resolve_output_path("05_router/evidence.json").read_text(
        encoding="utf-8"
    )
    assert (
        writer.start_stage(
            "router",
            "evidence",
            "05_router/evidence.json",
            operation_id="op-2",
        )
        is None
    )
    assert writer.resolve_output_path("05_router/evidence.json").read_text(
        encoding="utf-8"
    ) == before
    assert writer.result_summary()["status"] == "degraded"


def test_failed_index_write_is_repaired_by_same_path_retry(tmp_path):
    writer = _writer(tmp_path, "diagnostic-index-repair")
    original_write = writer._write_json_safely
    failed = {"once": True}

    def fail_index_once(path, payload):
        if path == writer.index_path and failed["once"]:
            failed["once"] = False
            writer._mark_degraded("模拟 index 写入失败", path=path)
            return False
        return original_write(path, payload)

    writer._write_json_safely = fail_index_once
    first = writer.start_stage("router", "evidence", "05_router/evidence.json")
    assert first is None
    assert writer.resolve_output_path("05_router/evidence.json").is_file()
    writer._write_json_safely = original_write
    repaired = writer.start_stage("router", "evidence", "05_router/evidence.json")
    assert repaired is not None and repaired.sequence == 1
    index = _read_json(writer.index_path)
    assert len(index["calls"]) == 1


def test_source_artifact_references_can_be_merged_only_before_terminal(tmp_path):
    writer = _writer(tmp_path, "diagnostic-artifacts")
    handle = writer.start_stage("router", "evidence", "05_router/batch_001/evidence.json")
    assert handle is not None
    assert writer.update_stage_source_artifacts(
        handle,
        {
            "base_input": {
                "relative_path": "working_memory/base.json",
                "sha256": "a" * 64,
                "media_type": "application/json",
                "size": 0,
            }
        },
    )
    assert writer.update_stage_source_artifacts(
        handle,
        {
            "manifest": {
                "relative_path": "working_memory/manifest.json",
                "sha256": "b" * 64,
                "media_type": "application/json",
                "size": 0,
            }
        },
    )
    stage_path = writer.resolve_output_path(
        "05_router/batch_001/evidence.json"
    )
    saved = _read_json(stage_path)
    assert set(saved["source_artifacts"]) == {"base_input", "manifest"}
    assert not (tmp_path / "working_memory").exists()

    assert writer.append_attempt(handle, {})
    assert writer.finish_stage(handle, "validated", final_output={"decision": "ready"})
    assert not writer.update_stage_source_artifacts(handle, {"response_files": []})


def test_immutable_view_is_created_once_and_cannot_be_replaced(tmp_path):
    writer = _writer(tmp_path, "diagnostic-immutable")
    path = "04_planning/batch_001/combination_001/workflow_versions/version_001.json"

    assert writer.write_immutable_view(
        path, {"schema_version": 1, "workflow_version": 1}
    )
    saved_path = writer.resolve_output_path(path)
    before = saved_path.read_text(encoding="utf-8")
    assert not writer.write_immutable_view(path, {"workflow_version": 2})
    assert saved_path.read_text(encoding="utf-8") == before
    assert writer.result_summary()["status"] == "degraded"


@pytest.mark.parametrize(
    "source_artifacts",
    [
        {"unexpected": _artifact("x.json")},
        {"base_input": _artifact("/absolute.json")},
        {"base_input": _artifact("a\\b.json")},
        {"base_input": _artifact("../escape.json")},
        {"request_files": _artifact("request.json")},
        {"response_files": [{"relative_path": "x.json", "sha256": "bad"}]},
    ],
)
def test_source_artifact_references_reject_invalid_aliases_and_paths(tmp_path, source_artifacts):
    writer = _writer(tmp_path, "diagnostic-invalid-artifacts")
    handle = writer.start_stage(
        "router",
        "evidence",
        "05_router/batch_001/evidence.json",
        source_artifacts=source_artifacts,
    )
    assert handle is None
    assert writer.result_summary()["status"] == "degraded"


def test_all_source_artifact_alias_arrays_are_saved_without_copying_files(tmp_path):
    writer = _writer(tmp_path, "diagnostic-artifact-arrays")
    refs = {
        "base_input": _artifact("working_memory/base.json"),
        "manifest": _artifact("working_memory/manifest.json"),
        "request_files": [_artifact("working_memory/request.json")],
        "response_files": [_artifact("working_memory/response.json")],
    }
    handle = writer.start_stage(
        "router",
        "evidence",
        "05_router/batch_001/evidence.json",
        source_artifacts=refs,
    )
    assert handle is not None
    assert writer.append_attempt(handle, {})
    assert writer.finish_stage(handle, "validated", final_output={"schema_version": 1})
    saved = _read_json(writer.resolve_output_path("05_router/batch_001/evidence.json"))
    assert set(saved["source_artifacts"]) == set(refs)
    assert not (tmp_path / "base.json").exists()


def test_summary_and_outcomes_are_atomic_json_updates(tmp_path):
    writer = _writer(tmp_path, "diagnostic-summary")
    assert writer.update_stage_summary(
        "03_hypothesis/batch_001/summary.json",
        {"schema_version": 1, "selected_hypotheses": []},
    )
    assert writer.update_hypothesis_outcomes(
        "03_hypothesis/batch_001/outcomes.json",
        {"schema_version": 1, "hypotheses": []},
    )
    root = writer.resolve_output_path("03_hypothesis/batch_001/summary.json").parent
    for path in (root / "summary.json", root / "outcomes.json"):
        assert json.loads(path.read_text(encoding="utf-8"))
        assert path.read_text(encoding="utf-8").endswith("\n")
        assert not list(path.parent.glob(f".{path.name}.*.tmp"))


def test_read_model_output_view_reads_only_existing_summary_or_outcomes(tmp_path):
    writer = _writer(tmp_path, "diagnostic-read-view")
    assert writer.read_model_output_view("03_hypothesis/summary.json") is None
    assert writer.result_summary()["status"] == "available"

    assert writer.update_summary("03_hypothesis/summary.json", {"batch_index": 1})
    assert writer.read_model_output_view("03_hypothesis/summary.json") == {
        "batch_index": 1
    }
    with pytest.raises(ValueError):
        writer.read_model_output_view("03_hypothesis/generation/query_001.json")


def test_write_failure_marks_degraded_without_raising_to_scientific_caller(tmp_path):
    writer = _writer(tmp_path, "diagnostic-degraded")

    def fail_atomic_write(path, payload):
        raise OSError("simulated disk failure")

    writer._atomic_write_json = fail_atomic_write
    assert not writer.update_stage_summary("01_retrieval/summary.json", {})
    summary = writer.result_summary()
    assert summary["status"] == "degraded"
    assert summary["error"] == {
        "failure_kind": "write_failed",
        "message": "模型输出展示文件写入失败",
    }


def test_external_observer_can_mark_degraded_without_persisting_error_text(tmp_path):
    writer = _writer(tmp_path, "diagnostic-observer")
    writer.mark_model_output_degraded("Authorization=do-not-persist")

    summary = writer.result_summary()
    task = _read_json(writer.task_path)
    assert summary["status"] == "degraded"
    assert "Authorization" not in json.dumps(summary, ensure_ascii=False)
    assert task["model_output_status"] == "degraded"


@pytest.mark.parametrize(
    "relative_path",
    [
        "../escape.json",
        str(Path(Path.cwd().anchor) / "tmp" / "escape.json"),
        "a\\b.json",
    ],
)
def test_stage_path_cannot_escape_model_output_root(tmp_path, relative_path):
    writer = _writer(tmp_path, "diagnostic-path")
    with pytest.raises(UnsafeModelOutputPath):
        writer.start_stage("query", "query_generation", relative_path)
    assert not (tmp_path / "escape.json").exists()
