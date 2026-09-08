"""作者假设导入与 research_task 应用入口的当前合同测试。"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from chemistry_multiagent.controllers.chemistry_multiagent_controller import (
    ResearchTaskApplication,
)
from chemistry_multiagent.controllers.investigation_v2_adapters import (
    AuthorHypothesisImportModuleAdapter,
)
from chemistry_multiagent.controllers.investigation_v2_module_controller import (
    ControllerAwaitingResearcherConfirmation,
    ResearchWorkflowController,
)
from chemistry_multiagent.investigation import author_hypothesis_import as author_import
from chemistry_multiagent.investigation.author_hypothesis_import import (
    load_case1_author_main_hypothesis,
    normalize_author_hypotheses,
)
from chemistry_multiagent.memory import ContractValidationError, ResearchTaskWorkingMemory


QUESTION = (
    "I plan to verify the mechanism of an asymmetric catalytic reaction using the reaction "
    "of p-nitrobenzaldehyde O=Cc1ccc(cc1)[N+](=O)[O-] with CC(=O)C under the catalysis "
    "of C(F)(F)(F)S(=O)(=O)O and N1CCC[C@H]1C(N2CCCC2), at 30°C in acetone solvent, "
    "to obtain the product CC(=O)C(O)c1ccc([N+](=O)[O-])cc1. Could you provide a "
    "specific approach to validate the reaction mechanism?"
)
FORBIDDEN_FINAL_VALUES = ("TS-2a", "TS-2b", "1.88", "88%")


def _normalized_author_hypotheses() -> list[dict[str, object]]:
    """从已审查的 Case 夹具构造通用作者输入，而不携带本机路径。"""

    fixture = load_case1_author_main_hypothesis(QUESTION)
    hypothesis = copy.deepcopy(dict(fixture.hypothesis))
    hypothesis_ref = hypothesis.get("hypothesis_id", hypothesis.get("hypothesis_ref"))
    assert isinstance(hypothesis_ref, str)
    return normalize_author_hypotheses(
        [
            {
                "hypothesis_ref": hypothesis_ref,
                "source_ref": fixture.fixture_id,
                "source_digest": fixture.fixture_sha256,
                "hypothesis": hypothesis,
                "provenance": copy.deepcopy(dict(fixture.provenance)),
            }
        ]
    )


def test_author_fixture_is_bound_to_question_context_and_hides_final_values() -> None:
    """固定夹具必须绑定题面，模型可见假设不得带作者最终答案。"""

    fixture = load_case1_author_main_hypothesis(QUESTION)
    hypothesis_text = json.dumps(fixture.hypothesis, ensure_ascii=False)
    assert all(marker not in hypothesis_text for marker in FORBIDDEN_FINAL_VALUES)
    with pytest.raises(ContractValidationError, match="题面不匹配"):
        load_case1_author_main_hypothesis(QUESTION + " drift")

    calculation_unit = copy.deepcopy(dict(fixture.calculation_unit))
    calculation_unit["charge"] = 0
    with pytest.raises(ContractValidationError, match="计算单元"):
        load_case1_author_main_hypothesis(
            QUESTION,
            {"calculation_unit": calculation_unit},
        )


def test_author_fixture_hash_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """夹具正文被篡改时必须在 provenance 校验边界停止。"""

    source = author_import._FIXTURE_PATH.read_text(encoding="utf-8")
    tampered = tmp_path / "tampered.json"
    tampered.write_text(
        source.replace("selection_status", "selection_status_tampered", 1),
        encoding="utf-8",
    )
    monkeypatch.setattr(author_import, "_FIXTURE_PATH", tampered)
    with pytest.raises(ContractValidationError, match="字段不完整|未知字段|selection"):
        load_case1_author_main_hypothesis(QUESTION)


def test_application_author_entry_freezes_canonical_input_without_old_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """通用作者入口只向 V2 任务装配传递科学问题和规范作者列表。"""

    application = object.__new__(ResearchTaskApplication)
    application.memory_enabled = True
    application._author_main_hypothesis_mode = False
    application.case_profile = None
    captured: dict[str, object] = {}

    def capture(**kwargs: object) -> dict[str, str]:
        captured.update(kwargs)
        return {"status": "open", "reason_code": "awaiting_researcher_confirmation"}

    monkeypatch.setattr(
        application,
        "_create_research_task_from_entry_strategy",
        capture,
        raising=False,
    )
    hypotheses = _normalized_author_hypotheses()
    result = application.create_author_hypothesis_task(QUESTION, hypotheses)

    assert result["status"] == "open"
    assert set(captured) == {"scientific_question", "author_hypotheses"}
    assert captured["scientific_question"] == QUESTION
    assert captured["author_hypotheses"] == hypotheses
    assert "run_mode" not in captured


def test_author_import_adapter_is_a_current_controller_boundary(tmp_path: Path) -> None:
    """作者入口跳过标准检索链，并在确认边界产生批次及逐假设记录。"""

    memory = ResearchTaskWorkingMemory(tmp_path / "research_task.sqlite3")
    hypotheses = _normalized_author_hypotheses()
    adapter = AuthorHypothesisImportModuleAdapter(
        controller=object(),
        task_root=tmp_path,
        working_memory=memory,
        task_input={
            "scientific_question": QUESTION,
            "author_hypotheses": hypotheses,
        },
        policy={"entry_strategy": "author_hypothesis_import"},
        dependencies={},
        task_config={},
    )
    controller = ResearchWorkflowController(
        memory,
        {"hypothesis-import": adapter},
        require_entry_strategy=True,
    )
    result = controller.run_research_task(
        "research-task-author",
        contract_version="investigation_v2.task.v2",
        input={
            "scientific_question": QUESTION,
            "author_hypotheses": hypotheses,
        },
        policy={"entry_strategy": "author_hypothesis_import"},
        dependencies={},
    )

    assert isinstance(result, ControllerAwaitingResearcherConfirmation)
    assert result.view.reason_code == "awaiting_researcher_confirmation"
    facts = memory.read_research_task_facts("research-task-author")
    assert [item.module_kind for item in facts.module_invocations] == [
        "hypothesis-import",
        "hypothesis-confirmation",
    ]
    records = memory.read_research_task_scientific_records("research-task-author")
    assert [record.scientific_record_type for record in records] == [
        "HypothesisBatchRecord",
        "SelectedHypothesisRecord",
    ]
    assert records[0].scientific_record_schema_version == "author-import.v2"
