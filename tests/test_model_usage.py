import json
from pathlib import Path

from chemistry_multiagent.utils.model_usage import summarize_task_token_usage


def _write_json(task_root: Path, relative_path: str, value: object) -> None:
    """在测试任务目录中写入一份受控 JSON 产物。"""

    target = task_root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _response_usage(prompt: int, completion: int, total: int) -> dict:
    """生成只保留计量所需字段的模型响应。"""

    return {
        "id": "受控响应",
        "choices": [{"message": {"content": "完成"}}],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
        },
    }


def test_task_usage_counts_paths_instead_of_deduplicating_equal_content(tmp_path):
    """内容相同但路径不同的两次响应必须分别计量。"""

    first_path = (
        "working_memory/hypothesis_generation_inputs/op/stages/2_query/"
        f"attempt_0_response_{'a' * 64}.json"
    )
    second_path = (
        "working_memory/hypothesis_generation_inputs/op/stages/3_hypothesis/"
        f"attempt_0_response_{'a' * 64}.json"
    )
    shared_response = _response_usage(10, 20, 30)
    _write_json(tmp_path, first_path, shared_response)
    _write_json(tmp_path, second_path, shared_response)

    summary = summarize_task_token_usage(tmp_path)

    assert summary["complete"] is True
    assert summary["response_artifact_count"] == 2
    assert summary["counted_response_count"] == 2
    assert summary["usage"] == {
        "prompt_tokens": 20,
        "completion_tokens": 40,
        "total_tokens": 60,
    }
    assert summary["known_usage"] == summary["usage"]
    assert summary["counted_response_paths"] == [first_path, second_path]
    assert summary["incomplete_responses"] == []
    assert summary["unmetered_transport_attempts"] == []


def test_task_usage_counts_retrieval_inner_model_calls_not_outer_wrapper(tmp_path):
    """Retrieval 聚合响应不计费，内部实际模型响应必须计费。"""

    outer_path = (
        "working_memory/hypothesis_generation_inputs/op/stages/1_retrieval/"
        f"attempt_0_response_{'b' * 64}.json"
    )
    inner_path = (
        "working_memory/hypothesis_generation_inputs/op/stages/1_retrieval/"
        f"model_calls/0_keyword/response_{'c' * 64}.json"
    )
    _write_json(tmp_path, inner_path, _response_usage(7, 5, 12))
    _write_json(
        tmp_path,
        outer_path,
        {
            "schema_version": 1,
            "result": {"keywords": ["benzaldehyde"]},
            "model_calls": [
                {
                    "stage": "keyword",
                    "transport_attempt_index": 0,
                    "request_ref": {"relative_path": "request.json"},
                    "response_ref": {"relative_path": inner_path},
                    "provider_trace_id": "trace-1",
                    "error_ref": None,
                    "validation_status": "valid",
                },
            ],
        },
    )

    summary = summarize_task_token_usage(tmp_path)

    assert summary["complete"] is True
    assert summary["response_artifact_count"] == 1
    assert summary["counted_response_count"] == 1
    assert summary["usage"] == {
        "prompt_tokens": 7,
        "completion_tokens": 5,
        "total_tokens": 12,
    }
    assert summary["counted_response_paths"] == [inner_path]
    assert summary["unmetered_transport_attempts"] == []


def test_task_usage_marks_missing_usage_as_incomplete_instead_of_zero(tmp_path):
    """只要有一条模型响应缺少 usage，就不能发布精确总量。"""

    complete_path = (
        "working_memory/router_inputs/router/stages/1_router/"
        f"attempt_0_response_{'d' * 64}.json"
    )
    missing_path = (
        "working_memory/workflow_generation_inputs/planner/stages/"
        f"1_workflow_generation/attempt_0_response_{'e' * 64}.json"
    )
    _write_json(tmp_path, complete_path, _response_usage(11, 9, 20))
    _write_json(
        tmp_path,
        missing_path,
        {
            "id": "没有计量字段的响应",
            "choices": [{"message": {"content": "完成"}}],
        },
    )

    summary = summarize_task_token_usage(tmp_path)

    assert summary["complete"] is False
    assert summary["response_artifact_count"] == 2
    assert summary["counted_response_count"] == 1
    assert summary["usage"] is None
    assert summary["known_usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 9,
        "total_tokens": 20,
    }
    assert summary["incomplete_responses"] == [
        {
            "relative_path": missing_path,
            "reason": "响应缺少完整 usage",
        }
    ]
    assert summary["unmetered_transport_attempts"] == []


def test_task_usage_marks_transport_without_response_as_unmetered(tmp_path):
    """没有响应回执的传输失败不能被当作精确零 token。"""

    manifest_path = (
        "working_memory/router_inputs/router-operation/"
        f"manifest_{'f' * 64}.json"
    )
    retrieval_wrapper_path = (
        "working_memory/hypothesis_generation_inputs/hypothesis-operation/"
        "stages/1_retrieval/"
        f"attempt_0_response_{'1' * 64}.json"
    )
    _write_json(
        tmp_path,
        manifest_path,
        {
            "schema_version": 1,
            "operation_instance_id": "router-operation",
            "operation": "router",
            "stages": [
                {
                    "stage_index": 1,
                    "attempts": [
                        {
                            "attempt_index": 0,
                            "validation_status": "transport_failed",
                        }
                    ],
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        retrieval_wrapper_path,
        {
            "schema_version": 1,
            "result": None,
            "model_calls": [
                {
                    "stage": "keyword",
                    "transport_attempt_index": 0,
                    "request_ref": {"relative_path": "request.json"},
                    "response_ref": None,
                    "provider_trace_id": None,
                    "error_ref": {"relative_path": "error.json"},
                    "validation_status": "transport_failed",
                }
            ],
        },
    )

    summary = summarize_task_token_usage(tmp_path)

    assert summary["complete"] is False
    assert summary["usage"] is None
    assert summary["known_usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    assert summary["unmetered_transport_attempts"] == [
        "router-operation/stage-1/attempt-0",
        (
            f"{retrieval_wrapper_path}#model_calls[0]"
        ),
    ]
