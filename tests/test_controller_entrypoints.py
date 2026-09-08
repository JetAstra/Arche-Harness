"""research_task 应用层与命令行公开动作的边界测试。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chemistry_multiagent.controllers import chemistry_multiagent_controller as module
from chemistry_multiagent.controllers.chemistry_multiagent_controller import (
    ResearchTaskApplication,
)


def _patch_application(monkeypatch: pytest.MonkeyPatch, application: type[Any]) -> None:
    """把 CLI 的应用装配替换为不访问模型或服务的记录替身。"""

    monkeypatch.setattr(module, "ResearchTaskApplication", application)
    monkeypatch.setattr(
        module,
        "build_expert_provider_bundle",
        lambda **_kwargs: "candidate-runtime",
    )


@pytest.mark.parametrize(
    "removed_arguments",
    [
        ["--toolpool", "tools.json"],
        ["--gaussian-execution-mode", "local_shell"],
        ["--gaussian-command", "/opt/g16/g16"],
        ["--expert-model-path", "/tmp/legacy-model"],
        ["--expert-backend", "local_hf"],
        ["--disable-expert-review"],
    ],
)
def test_removed_runtime_options_are_not_research_task_actions(
    removed_arguments: list[str],
) -> None:
    """旧运行时覆盖参数不能重新成为 research_task 的入口协议。"""

    with pytest.raises(SystemExit):
        module.run_from_args(removed_arguments)


def test_application_exposes_only_formal_research_task_actions() -> None:
    """应用层公开的是 research_task 动作集合。"""

    expected = {
        "create_standard_research_task",
        "create_author_hypothesis_task",
        "create_top5_confirmation_task",
        "get_hypothesis_confirmation",
        "submit_hypothesis_confirmation_and_continue",
        "resume_research_task",
    }
    assert all(callable(getattr(ResearchTaskApplication, name, None)) for name in expected)
    assert not hasattr(ResearchTaskApplication, "run_task")
    assert not hasattr(ResearchTaskApplication, "run_investigation_and_experiment_task")


def test_resume_cli_routes_to_research_task_application(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """resume-research-task 只传递 research_task_id。"""

    calls: list[tuple[str, Any]] = []

    class FakeApplication:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(("init", kwargs))

        def resume_research_task(self, research_task_id: str, **kwargs: Any) -> dict[str, Any]:
            calls.append(("resume_research_task", (research_task_id, kwargs)))
            return {"research_task_id": research_task_id, "status": "running"}

    _patch_application(monkeypatch, FakeApplication)
    result = module.run_from_args(
        [
            "resume-research-task",
            "research-task-1",
            "--disable-expert-model",
            "--memory-db",
            "/tmp/research-task.sqlite3",
        ]
    )

    assert result == {"research_task_id": "research-task-1", "status": "running"}
    assert calls[0][0] == "init"
    assert calls[0][1]["memory_enabled"] is True
    assert calls[0][1]["memory_db_path"] == "/tmp/research-task.sqlite3"
    assert calls[1] == ("resume_research_task", ("research-task-1", {}))
    assert json.loads(capsys.readouterr().out) == result


@pytest.mark.parametrize(
    "action_arguments",
    [
        ["--authorize-legacy-waiting-resume"],
        ["--authorize-incomplete-v2-waiting-resume"],
    ],
)
def test_migration_authorization_requires_resume_action(
    action_arguments: list[str],
) -> None:
    """迁移边界参数不能脱离正式 research_task 恢复动作使用。"""

    with pytest.raises(SystemExit):
        module.run_from_args(action_arguments)


def test_create_standard_cli_routes_canonical_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """标准创建动作把问题、论文策略和预算交给应用层。"""

    calls: list[dict[str, Any]] = []

    class FakeApplication:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def create_standard_research_task(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs)
            return {
                "research_task_id": "research-task-1",
                "status": "open",
            }

    _patch_application(monkeypatch, FakeApplication)
    module.run_from_args(
        [
            "create-standard-research-task",
            "--question",
            "验证标准 research_task 入口。",
            "--no-search",
            "--disable-expert-model",
            "--num-queries",
            "2",
            "--num-hypotheses-per-query",
            "4",
            "--max-parallel-investigations",
            "3",
            "--max-investigation-repairs-per-combination",
            "2",
            "--max-hypothesis-batches",
            "2",
        ]
    )

    assert calls == [
        {
            "scientific_question": "验证标准 research_task 入口。",
            "num_queries": 2,
            "num_hypotheses_per_query": 4,
            "top_n_hypotheses": 5,
            "pdf_dir": "papers",
            "index_dir": "index",
            "search_papers": False,
            "paper_search": {
                "enabled": False,
                "provider_strategy": "pypaperbot_scholar",
                "target_papers": 50,
                "manifest_path": None,
                "resume_state_path": None,
                "response_cache_dir": None,
                "timeout_seconds": 180.0,
                "candidate_page_size": 100,
                "openalex_per_page": 100,
                "keyword_delay_seconds": 45.0,
                "keyword_jitter_seconds": 45.0,
                "download_delay_seconds": 1.0,
                "scholar_max_pages": 3,
                "scholar_results_per_page": 10,
                "scholar_crossref_max_candidates": 3,
                "scholar_request_delay_seconds": 15.0,
                "scholar_request_jitter_seconds": 15.0,
                "scholar_initial_delay_seconds": 10.0,
                "scholar_initial_jitter_seconds": 20.0,
                "scholar_backoff_base_seconds": 900.0,
                "scholar_max_backoff_seconds": 3600.0,
                "scholar_max_consecutive_rate_limits": 1,
            },
            "max_parallel_investigations": 3,
            "max_investigation_repairs_per_combination": 2,
            "max_hypothesis_batches": 2,
            "supplementary_scientific_input": None,
        }
    ]


def test_create_top5_cli_routes_confirmation_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Top 5 创建动作进入独立的确认任务入口。"""

    calls: list[dict[str, Any]] = []

    class FakeApplication:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def create_top5_confirmation_task(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs)
            return {"research_task_id": "top5-task", "status": "open"}

    _patch_application(monkeypatch, FakeApplication)
    module.run_from_args(
        [
            "create-top5-confirmation-task",
            "--question",
            "验证 Top 5 确认入口。",
            "--no-search",
            "--disable-expert-model",
        ]
    )

    assert calls[0]["scientific_question"] == "验证 Top 5 确认入口。"
    assert calls[0]["top_n_hypotheses"] == 5
    assert calls[0]["search_papers"] is False


def test_create_author_cli_loads_structured_hypotheses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """作者假设文件在 CLI 适配层解析后，以结构化内容进入应用。"""

    author_file = tmp_path / "author-hypotheses.json"
    author_file.write_text(
        json.dumps(
            [
                {
                    "hypothesis_ref": "author-h1",
                    "source_ref": "author-source-1",
                    "source_digest": "a" * 64,
                    "hypothesis": {
                        "hypothesis_id": "author-h1",
                        "strategy_name": "author strategy",
                        "reasoning": "author reasoning",
                    },
                    "provenance": {"kind": "researcher"},
                }
            ]
        ),
        encoding="utf-8",
    )
    calls: list[dict[str, Any]] = []

    class FakeApplication:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def create_author_hypothesis_task(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs)
            return {"research_task_id": "author-task", "status": "open"}

    _patch_application(monkeypatch, FakeApplication)
    module.run_from_args(
        [
            "create-author-hypothesis-task",
            "--question",
            "验证作者假设入口。",
            "--author-hypotheses-json",
            str(author_file),
            "--disable-expert-model",
        ]
    )

    assert calls[0]["scientific_question"] == "验证作者假设入口。"
    assert calls[0]["author_hypotheses"][0]["hypothesis_ref"] == "author-h1"
    assert "path" not in calls[0]["author_hypotheses"][0]


def test_get_confirmation_cli_is_read_only_application_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """确认视图读取动作不被误路由为恢复或新建任务。"""

    calls: list[str] = []

    class FakeApplication:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def get_hypothesis_confirmation(self, research_task_id: str) -> dict[str, Any]:
            calls.append(research_task_id)
            return {
                "research_task_id": research_task_id,
                "status": "open",
                "candidates": [],
            }

    _patch_application(monkeypatch, FakeApplication)
    result = module.run_from_args(
        [
            "get-hypothesis-confirmation",
            "research-task-1",
            "--disable-expert-model",
        ]
    )

    assert calls == ["research-task-1"]
    assert result["status"] == "open"


def test_submit_confirmation_cli_passes_canonical_request_and_streams(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """确认提交动作按 canonical 字段调用应用，并把材料作为流传入。"""

    request_file = tmp_path / "confirmation.json"
    material_file = tmp_path / "material.txt"
    material_file.write_bytes(b"material")
    request_file.write_text(
        json.dumps(
            {
                "confirmation_id": "confirmation-1",
                "expected_research_task_sequence": 19,
                "decisions": [],
                "files": {
                    "upload-1": {
                        "path": material_file.name,
                        "filename": "material.txt",
                        "media_type": "text/plain",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    calls: list[dict[str, Any]] = []
    received_material: list[bytes] = []

    class FakeApplication:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def submit_hypothesis_confirmation_and_continue(
            self,
            research_task_id: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            calls.append({"research_task_id": research_task_id, **kwargs})
            received_material.append(kwargs["files"]["upload-1"].content.read())
            return {"research_task_id": research_task_id, "status": "running"}

    _patch_application(monkeypatch, FakeApplication)
    result = module.run_from_args(
        [
            "submit-hypothesis-confirmation-and-continue",
            "research-task-1",
            "--confirmation-request-json",
            str(request_file),
            "--disable-expert-model",
        ]
    )

    assert result["status"] == "running"
    assert calls[0]["research_task_id"] == "research-task-1"
    assert calls[0]["confirmation_id"] == "confirmation-1"
    assert calls[0]["expected_research_task_sequence"] == 19
    assert calls[0]["decisions"] == []
    upload = calls[0]["files"]["upload-1"]
    assert upload.filename == "material.txt"
    assert upload.media_type == "text/plain"
    assert received_material == [b"material"]


@pytest.mark.parametrize(
    ("status", "expected_exit_code"),
    [
        ("failed", 1),
        ("retrieval_incomplete", 1),
        ("succeeded", 0),
        ("stopped_no_supported_hypothesis", 0),
    ],
)
def test_main_maps_research_task_status_to_process_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    expected_exit_code: int,
) -> None:
    """只有正式失败状态映射为非零命令行退出码。"""

    monkeypatch.setattr(
        module,
        "run_from_args",
        lambda _argv: {"status": status},
    )

    assert module.main(["unused-by-stub"]) == expected_exit_code
