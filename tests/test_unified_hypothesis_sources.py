"""统一作者假设输入与 research_task 公开动作的合同回归测试。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from chemistry_multiagent.controllers import chemistry_multiagent_controller as module
from chemistry_multiagent.controllers.chemistry_multiagent_controller import (
    ResearchTaskApplication,
)
from chemistry_multiagent.controllers.investigation_v2_adapters import (
    AuthorHypothesisImportModuleAdapter,
)
from chemistry_multiagent.controllers.investigation_v2_module_controller import (
    ControllerAwaitingResearcherConfirmation,
    ModuleInvocationRequest,
    ModuleSuspended,
    ResearchWorkflowController,
)
from chemistry_multiagent.controllers.investigation_v2_production import (
    DEFAULT_RESEARCH_MATERIAL_POLICY,
    ResearchTaskProductionAssembly,
    V2_TASK_CONTRACT_VERSION,
)
from chemistry_multiagent.investigation.author_hypothesis_import import (
    normalize_author_hypotheses,
)
from chemistry_multiagent.memory import ResearchTaskWorkingMemory


def _hypothesis(ref: str = "author-h1") -> dict[str, Any]:
    """返回一条不含材料和本机路径的结构化作者假设。"""

    return {
        "hypothesis_ref": ref,
        "source_ref": "author-paper-1",
        "source_digest": "a" * 64,
        "hypothesis": {
            "hypothesis_id": ref,
            "strategy_name": "作者提出的反应路径",
            "reasoning": "作者认为该路径可以回答原始科学问题。",
            "supporting_observation": "独立来源支持该机理。",
        },
        "provenance": {
            "author_source_document_id": "paper-1",
            "pages": [3, 4],
        },
    }


def _application() -> ResearchTaskApplication:
    """构造不触发模型或服务的最小 Application。"""

    application = object.__new__(ResearchTaskApplication)
    application.memory_enabled = True
    application._author_main_hypothesis_mode = False
    application.case_profile = None
    return application


def test_author_entry_passes_structured_input_without_case_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """作者直入只传问题和规范作者列表，不注入旧 run mode 或 Case 开关。"""

    calls: list[dict[str, Any]] = []
    application = _application()

    def capture(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"status": "open", "reason_code": "awaiting_researcher_confirmation"}

    monkeypatch.setattr(
        application,
        "_create_research_task_from_entry_strategy",
        capture,
        raising=False,
    )
    result = application.create_author_hypothesis_task(
        "验证作者提出的反应机理",
        [_hypothesis()],
    )

    assert result["status"] == "open"
    assert calls == [
        {
            "scientific_question": "验证作者提出的反应机理",
            "author_hypotheses": [_hypothesis()],
        }
    ]


@pytest.mark.parametrize(
    "invalid",
    [
        [],
        [_hypothesis()] * 6,
        [{**_hypothesis(), "materials": []}],
        [{**_hypothesis(), "source_digest": "not-a-sha"}],
        [{**_hypothesis(), "source_ref": "/tmp/author-paper.json"}],
        [{**_hypothesis(), "hypothesis": {"hypothesis_id": "author-h1"}}],
        [{**_hypothesis(), "provenance": "/tmp/provenance.json"}],
    ],
)
def test_author_hypothesis_contract_rejects_ambiguous_input(
    invalid: list[dict[str, Any]],
) -> None:
    """作者来源只能是 1–5 条结构化记录，不能携带材料或调用方路径。"""

    with pytest.raises(ValueError):
        normalize_author_hypotheses(invalid)


def test_production_assembly_persists_current_v2_contract_and_policy(
    tmp_path: Path,
) -> None:
    """生产装配把 V2 合同和材料策略写入同一 research_task Facts。"""

    class SuspendedModule:
        def start(self, request: ModuleInvocationRequest) -> ModuleSuspended:
            del request
            return ModuleSuspended("provider_unavailable")

        def resume(self, request: ModuleInvocationRequest) -> ModuleSuspended:
            return self.start(request)

    class _Application:
        _run_storage_root = tmp_path
        memory_db_path = None
        _memory_db_explicit = False

    assembly = ResearchTaskProductionAssembly(
        _Application(),
        module_adapters={"retrieval": SuspendedModule()},
    )
    result = assembly.create_research_task(
        research_task_id="research-task-v2-contract",
        scientific_question="验证版本合同",
        task_config={"top_n_hypotheses": 5},
        task_input={"scientific_question": "验证版本合同"},
        policy={
            "entry_strategy": "standard_hypothesis_generation",
            "stage_architecture_version": "investigation_v2",
        },
        dependencies={},
    )

    assert result["research_task_id"] == "research-task-v2-contract"
    assert result["status"] == "open"
    db_path = (
        tmp_path
        / "outputs"
        / "research_tasks"
        / "research-task-v2-contract"
        / "internal"
        / "state"
        / "arche_memory.sqlite3"
    )
    facts = ResearchTaskWorkingMemory(db_path).read_research_task_facts(
        "research-task-v2-contract"
    )
    assert facts.research_task.contract_version == V2_TASK_CONTRACT_VERSION
    assert facts.research_task.policy["research_material_policy"] == (
        DEFAULT_RESEARCH_MATERIAL_POLICY
    )
    assert "task_contract_version" not in facts.research_task.policy


def test_production_author_assembly_exposes_current_confirmation_view(tmp_path: Path) -> None:
    """生产公开结果返回 formal 候选字段，而不是把确认状态视为未知。"""

    hypotheses = [_hypothesis()]
    memory = ResearchTaskWorkingMemory(tmp_path / "unused.sqlite3")
    adapter = AuthorHypothesisImportModuleAdapter(
        controller=object(),
        task_root=tmp_path,
        working_memory=memory,
        task_input={
            "scientific_question": "选择待执行假设",
            "author_hypotheses": normalize_author_hypotheses(hypotheses),
        },
        policy={"entry_strategy": "author_hypothesis_import"},
        dependencies={},
        task_config={},
    )

    class _Application:
        _run_storage_root = tmp_path
        memory_db_path = None
        _memory_db_explicit = False

    result = ResearchTaskProductionAssembly(
        _Application(),
        module_adapters={"hypothesis-import": adapter},
    ).create_research_task(
        research_task_id="research-task-awaiting-confirmation",
        scientific_question="选择待执行假设",
        task_config={"top_n_hypotheses": 1},
        task_input={
            "scientific_question": "选择待执行假设",
            "author_hypotheses": normalize_author_hypotheses(hypotheses),
        },
        policy={
            "entry_strategy": "author_hypothesis_import",
            "stage_architecture_version": "investigation_v2",
        },
        dependencies={},
    )

    assert result["status"] == "open"
    assert result["reason_code"] == "awaiting_researcher_confirmation"
    assert result["research_task_id"] == "research-task-awaiting-confirmation"
    assert result["hypothesis_batch_record_id"]
    candidate = result["candidates"][0]
    assert set(candidate) == {
        "hypothesis_candidate_record_id",
        "rank",
        "hypothesis",
    }
    assert not {"candidate_ref", "task_id", "module_run_id"}.intersection(result)


def test_cli_author_json_uses_same_public_application_action(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CLI 文件适配器只负责解析 JSON，最终调用统一作者动作。"""

    input_path = tmp_path / "author-hypotheses.json"
    input_path.write_text(json.dumps([_hypothesis()], ensure_ascii=False), encoding="utf-8")
    calls: list[dict[str, Any]] = []

    class _Application:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def create_author_hypothesis_task(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs)
            return {"status": "open", "research_task_id": "research-task-author"}

    monkeypatch.setattr(module, "ResearchTaskApplication", _Application)
    result = module.run_from_args(
        [
            "create-author-hypothesis-task",
            "--question",
            "验证作者提出的反应机理",
            "--author-hypotheses-json",
            str(input_path),
            "--disable-expert-model",
        ]
    )

    assert result == {"status": "open", "research_task_id": "research-task-author"}
    assert calls == [
        {
            "scientific_question": "验证作者提出的反应机理",
            "author_hypotheses": [_hypothesis()],
        }
    ]


def test_cli_confirmation_streams_files_to_current_application_action(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CLI 只在入口读取本地路径，Application 收到正式确认字段和文件流。"""

    material_path = tmp_path / "matrix-elements.json"
    material_path.write_bytes(b'{"atoms":[1,2,3,4]}')
    request_path = tmp_path / "confirmation.json"
    decisions = [
        {
            "hypothesis_candidate_record_id": "hypothesis-1",
            "decision": "execute",
            "hypothesis": {"hypothesis_id": "hypothesis-1", "claim": "claim"},
            "guidance": "追踪二面角变化",
            "material_upload_refs": ["matrix-elements"],
        }
    ]
    request_path.write_text(
        json.dumps(
            {
                "confirmation_id": "confirmation-cli-1",
                "expected_research_task_sequence": 17,
                "decisions": decisions,
                "files": {
                    "matrix-elements": {
                        "path": material_path.name,
                        "filename": "matrix-elements.json",
                        "media_type": "application/json",
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    calls: list[dict[str, Any]] = []

    class _Application:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def submit_hypothesis_confirmation_and_continue(
            self, research_task_id: str, **kwargs: Any
        ) -> dict[str, Any]:
            upload = kwargs["files"]["matrix-elements"]
            calls.append(
                {
                    "research_task_id": research_task_id,
                    "confirmation_id": kwargs["confirmation_id"],
                    "expected_research_task_sequence": kwargs[
                        "expected_research_task_sequence"
                    ],
                    "decisions": kwargs["decisions"],
                    "filename": upload.filename,
                    "media_type": upload.media_type,
                    "content": upload.content.read(),
                }
            )
            return {"status": "open", "research_task_id": research_task_id}

    monkeypatch.setattr(module, "ResearchTaskApplication", _Application)
    result = module.run_from_args(
        [
            "submit-hypothesis-confirmation-and-continue",
            "research-task-confirm-cli",
            "--confirmation-request-json",
            str(request_path),
            "--disable-expert-model",
        ]
    )

    assert result == {"status": "open", "research_task_id": "research-task-confirm-cli"}
    assert calls == [
        {
            "research_task_id": "research-task-confirm-cli",
            "confirmation_id": "confirmation-cli-1",
            "expected_research_task_sequence": 17,
            "decisions": decisions,
            "filename": "matrix-elements.json",
            "media_type": "application/json",
            "content": b'{"atoms":[1,2,3,4]}',
        }
    ]
    assert str(material_path) not in repr(calls)


def test_cli_confirmation_rejects_symlink_material_before_application_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CLI 材料适配器拒绝符号链接，不能把它交给确认动作。"""

    target = tmp_path / "target.json"
    target.write_bytes(b"{}")
    material_path = tmp_path / "material-link.json"
    material_path.symlink_to(target)
    request_path = tmp_path / "confirmation.json"
    request_path.write_text(
        json.dumps(
            {
                "confirmation_id": "confirmation-cli-symlink",
                "expected_research_task_sequence": 3,
                "decisions": [],
                "files": {
                    "material": {
                        "path": material_path.name,
                        "filename": "material.json",
                        "media_type": "application/json",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    class _Application:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def submit_hypothesis_confirmation_and_continue(self, *_args: Any, **_kwargs: Any):
            raise AssertionError("符号链接材料不能进入 Application")

    monkeypatch.setattr(module, "ResearchTaskApplication", _Application)

    with pytest.raises(SystemExit):
        module.run_from_args(
            [
                "submit-hypothesis-confirmation-and-continue",
                "research-task-confirm-cli",
                "--confirmation-request-json",
                str(request_path),
                "--disable-expert-model",
            ]
        )


def test_cli_confirmation_rejects_special_material_before_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CLI 在打开上传流前拒绝 FIFO 等特殊文件，避免阻塞。"""

    material_path = tmp_path / "material.fifo"
    os.mkfifo(material_path)
    request_path = tmp_path / "confirmation.json"
    request_path.write_text(
        json.dumps(
            {
                "confirmation_id": "confirmation-cli-fifo",
                "expected_research_task_sequence": 3,
                "decisions": [],
                "files": {
                    "material": {
                        "path": material_path.name,
                        "filename": "material.bin",
                        "media_type": "application/octet-stream",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    class _Application:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def submit_hypothesis_confirmation_and_continue(self, *_args: Any, **_kwargs: Any):
            raise AssertionError("特殊文件不能进入 Application")

    monkeypatch.setattr(module, "ResearchTaskApplication", _Application)

    with pytest.raises(SystemExit):
        module.run_from_args(
            [
                "submit-hypothesis-confirmation-and-continue",
                "research-task-confirm-cli",
                "--confirmation-request-json",
                str(request_path),
                "--disable-expert-model",
            ]
        )
