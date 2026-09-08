"""Retrieval 模型展示旁路的独立测试。"""

from __future__ import annotations

import json
import inspect
from pathlib import Path
from typing import Any

import fitz

from chemistry_multiagent.agents.retrieval_agent import RetrievalAgent
from tests.fakes.retrieval import make_test_retrieval_agent
from chemistry_multiagent.utils.model_run_outputs import ModelRunOutputWriter
from chemistry_multiagent.utils.model_run_retrieval import (
    RetrievalModelOutputRecorder,
)


QUESTION = (
    "Design a workflow to convert benzaldehyde (SMILES: O=Cc1ccccc1) to SDF, "
    "generate Gaussian input, optimize geometry, run frequency analysis, and "
    "plot IR and Raman spectra."
)


def _context() -> dict[str, Any]:
    return {
        "reaction_type": "unknown",
        "mechanistic_goal": "unknown",
        "suspected_job_types": ["freq", "opt"],
        "solvent": None,
        "temperature": None,
        "charge": None,
        "multiplicity": None,
        "species_roles": [],
        "candidate_elements": ["C", "O"],
        "needs_ts": None,
        "needs_irc": None,
        "needs_excited_state": None,
        "evidence_gaps": [],
    }


def _freeze_no_search_pdf(tmp_path: Path) -> Path:
    """为 no-search 单测创建一份真实 PDF 与冻结 manifest。"""

    pdf_dir = tmp_path / "papers"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / "paper.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "frozen retrieval fixture")
    document.save(pdf_path)
    document.close()
    manifest_agent = make_test_retrieval_agent()
    manifest_agent._write_paper_search_manifest(
        question=QUESTION,
        keywords=[],
        pdf_dir=str(pdf_dir),
        manifest_path=str(pdf_dir / ".paper-search-manifest.json"),
        target_papers=1,
        search_config={"target_papers": 1, "provider_strategy": "frozen_fixture"},
        keyword_results=[],
        keyword_source="override",
    )
    return pdf_dir


def _envelope(content: str, *, trace_id: str = "retrieval-trace") -> dict[str, Any]:
    return {
        "id": trace_id,
        "model": "test-model",
        "provider": "openai_compatible",
        "authorization": "must-not-be-displayed",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
            "input_tokens": 11,
            "output_tokens": 7,
        },
    }


def _stage_path(root: Path) -> Path:
    return (
        root
        / "outputs"
        / "research_tasks"
        / "diagnostic-retrieval"
        / "internal"
        / "audit"
        / "model_calls"
        / "01_retrieval"
        / "summarize_retrieval.json"
    )


def _make_recorder(tmp_path: Path) -> RetrievalModelOutputRecorder:
    writer = ModelRunOutputWriter(tmp_path, "diagnostic-retrieval")
    return RetrievalModelOutputRecorder(
        writer,
        {
            "scientific_question": QUESTION,
            "pdf_dir": "papers",
            "index_dir": "index",
            "search_papers": False,
            "authorization": "must-not-be-displayed",
        },
        "retrieval-operation-1",
    )


def _read_stage(tmp_path: Path) -> dict[str, Any]:
    return json.loads(_stage_path(tmp_path).read_text(encoding="utf-8"))


def _finish(recorder: RetrievalModelOutputRecorder, result: dict[str, Any]) -> None:
    """模拟 Controller 在 Retrieval 返回后关闭展示阶段。"""

    recorder.finish_retrieval(result)


def test_no_model_call_does_not_create_retrieval_stage(tmp_path: Path, monkeypatch) -> None:
    """没有实际模型调用时不应创建 01_retrieval 阶段文件。"""

    recorder = _make_recorder(tmp_path)
    agent = make_test_retrieval_agent(retrieval_provider=None)
    monkeypatch.setattr(agent, "extract_keywords", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(agent, "build_index", lambda *_args, **_kwargs: None)

    result = agent.process_question(
        QUESTION,
        pdf_dir="papers",
        index_dir="index",
        search_papers=False,
        frozen_chemistry_context=_context(),
        receipt_sink=recorder,
    )
    _finish(recorder, result)

    assert result["chemistry_context"] == _context()
    assert not _stage_path(tmp_path).exists()


def test_retrieval_substages_and_protocol_retries_share_one_stage_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """多个子阶段与化学上下文重答按实际顺序写进同一个文件。"""

    responses = [
        _envelope("```json\n{\n```", trace_id="context-invalid-1"),
        _envelope("```json\n{\n```", trace_id="context-invalid-2"),
        _envelope(json.dumps(_context(), ensure_ascii=False), trace_id="context-ok"),
        _envelope("文献问答结果", trace_id="qa-ok"),
        _envelope("文献综述结果", trace_id="review-ok"),
    ]

    def provider(_messages):
        return responses.pop(0)

    recorder = _make_recorder(tmp_path)
    pdf_dir = _freeze_no_search_pdf(tmp_path)
    agent = make_test_retrieval_agent(retrieval_provider=provider)
    monkeypatch.setattr(
        agent,
        "build_index",
        lambda *_args, **_kwargs: setattr(
            agent,
            "documents",
            ["文献片段"],
        ),
    )
    monkeypatch.setattr(
        agent,
        "search",
        lambda *_args, **_kwargs: [
            {"text": "文献片段", "metadata": {"source": "paper.pdf", "section": "methods"}}
        ],
    )

    result = agent.process_question(
        QUESTION,
        pdf_dir=str(pdf_dir),
        index_dir=str(tmp_path / "index"),
        search_papers=False,
        receipt_sink=recorder,
    )
    _finish(recorder, result)

    stage = _read_stage(tmp_path)
    assert stage["status"] == "validated"
    assert stage["final_output"] == result
    assert [item["attempt_index"] for item in stage["attempts"]] == list(range(5))
    assert [
        item["request"]["catalog_key"] for item in stage["attempts"]
    ] == [
        "retrieval.chemistry_context",
        "retrieval.chemistry_context",
        "retrieval.chemistry_context",
        "retrieval.document_qa",
        "retrieval.literature_review",
    ]
    assert all(
        set(item["request"]["response_contract"])
        == {"validation_kind", "schema", "domain_validator_id"}
        for item in stage["attempts"]
    )
    assert [item["validation"]["status"] for item in stage["attempts"]] == [
        "invalid",
        "invalid",
        "valid",
        "valid",
        "valid",
    ]
    assert all(item["started_at"] and item["finished_at"] for item in stage["attempts"])
    assert all(item["duration_ms"] >= 0 for item in stage["attempts"])
    assert all(
        "authorization" not in json.dumps(item, ensure_ascii=False).lower()
        for item in stage["attempts"]
    )
    assert responses == []


def test_transport_retry_exhaustion_is_one_display_attempt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """内部网络重试耗尽只展示一个 transport_failed 逻辑回答。"""

    calls = []

    def provider(request):
        calls.append(request)
        raise ConnectionError("secret provider network detail")

    recorder = _make_recorder(tmp_path)
    pdf_dir = _freeze_no_search_pdf(tmp_path)
    monkeypatch.setattr(
        "chemistry_multiagent.agents.retrieval_agent.call_compiled_model_request",
        provider,
    )
    agent = make_test_retrieval_agent(retrieval_provider=None)
    monkeypatch.setattr(
        agent,
        "build_index",
        lambda *_args, **_kwargs: setattr(agent, "documents", ["文献片段"]),
    )
    monkeypatch.setattr(
        agent,
        "search",
        lambda *_args, **_kwargs: [
            {"text": "文献片段", "metadata": {"section": "methods"}}
        ],
    )
    result = agent.process_question(
        QUESTION,
        pdf_dir=str(pdf_dir),
        index_dir=str(tmp_path / "index"),
        search_papers=False,
        frozen_chemistry_context=_context(),
        receipt_sink=recorder,
    )
    _finish(recorder, result)

    stage = _read_stage(tmp_path)
    assert len(calls) == 6
    assert result["error"]
    assert stage["status"] == "failed"
    assert len(stage["attempts"]) == 1
    attempt = stage["attempts"][0]
    assert attempt["validation"]["status"] == "transport_failed"
    assert attempt["validation"]["failure_kind"] == "transport"
    assert attempt["validation"]["message"] == "模型服务请求失败"
    assert attempt["raw_response"] is None
    assert "secret provider network detail" not in json.dumps(stage, ensure_ascii=False)


def test_provider_exception_without_receipt_still_finishes_failed_stage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """provider 未发 receipt 时，已开始调用也要生成安全 transport 失败记录。"""

    recorder = _make_recorder(tmp_path)
    pdf_dir = _freeze_no_search_pdf(tmp_path)
    agent = make_test_retrieval_agent(retrieval_provider=None)
    monkeypatch.setattr(
        agent,
        "build_index",
        lambda *_args, **_kwargs: setattr(agent, "documents", ["文献片段"]),
    )
    monkeypatch.setattr(
        agent,
        "search",
        lambda *_args, **_kwargs: [
            {"text": "文献片段", "metadata": {"section": "methods"}}
        ],
    )

    def raise_without_receipt(*_args, **_kwargs):
        raise RuntimeError("secret provider exception")

    monkeypatch.setattr(agent, "_call_retrieval_provider", raise_without_receipt)
    result = agent.process_question(
        QUESTION,
        pdf_dir=str(pdf_dir),
        index_dir=str(tmp_path / "index"),
        search_papers=False,
        frozen_chemistry_context=_context(),
        receipt_sink=recorder,
    )
    _finish(recorder, result)

    stage = _read_stage(tmp_path)
    assert result["error"]
    assert stage["status"] == "failed"
    assert len(stage["attempts"]) == 1
    attempt = stage["attempts"][0]
    assert attempt["validation"] == {
        "status": "transport_failed",
        "failure_kind": "transport",
        "message": "模型服务请求失败",
    }
    assert attempt["raw_response"] is None
    assert attempt["provider"] is None
    assert attempt["model"] is None
    assert attempt["provider_trace_id"] is None
    assert attempt["usage"] is None
    assert set(attempt["request"]["response_contract"]) == {
        "validation_kind",
        "schema",
        "domain_validator_id",
    }
    assert "secret provider exception" not in json.dumps(stage, ensure_ascii=False)


class _BrokenWriter:
    def start_stage(self, *args, **kwargs):
        raise OSError("writer failure")

    def append_attempt(self, *args, **kwargs):
        raise OSError("writer failure")

    def finish_stage(self, *args, **kwargs):
        raise OSError("writer failure")

    def mark_model_output_degraded(self, *args, **kwargs):
        raise OSError("writer failure")


def test_recorder_write_failure_does_not_change_retrieval_result(
    monkeypatch, tmp_path: Path
) -> None:
    """展示 writer 失败时，Retrieval 仍返回原有科学结果。"""

    def provider(_messages):
        return _envelope("固定文献回答")

    pdf_dir = _freeze_no_search_pdf(tmp_path)
    agent = make_test_retrieval_agent(retrieval_provider=provider)
    monkeypatch.setattr(
        agent,
        "build_index",
        lambda *_args, **_kwargs: setattr(agent, "documents", ["文献片段"]),
    )
    monkeypatch.setattr(
        agent,
        "search",
        lambda *_args, **_kwargs: [
            {"text": "文献片段", "metadata": {"section": "methods"}}
        ],
    )
    recorder = RetrievalModelOutputRecorder(
        _BrokenWriter(),
        {"scientific_question": QUESTION, "search_papers": False},
        "retrieval-operation-broken-writer",
    )
    result = agent.process_question(
        QUESTION,
        pdf_dir=str(pdf_dir),
        index_dir=str(tmp_path / "index"),
        search_papers=False,
        frozen_chemistry_context=_context(),
        receipt_sink=recorder,
    )
    _finish(recorder, result)

    assert result["answer"] == "固定文献回答"
    assert result["literature_review"] == "固定文献回答"
    assert recorder.degraded is True


def test_ordinary_receipt_sink_keeps_existing_receipts_unchanged(
    monkeypatch, tmp_path: Path
) -> None:
    """普通 callable receipt sink 不应被展示旁路钩子改变。"""

    receipts: list[dict[str, Any]] = []

    def provider(_messages):
        return _envelope("固定文献回答")

    pdf_dir = _freeze_no_search_pdf(tmp_path)
    agent = make_test_retrieval_agent(retrieval_provider=provider)
    monkeypatch.setattr(
        agent,
        "build_index",
        lambda *_args, **_kwargs: setattr(agent, "documents", ["文献片段"]),
    )
    monkeypatch.setattr(
        agent,
        "search",
        lambda *_args, **_kwargs: [
            {"text": "文献片段", "metadata": {"section": "methods"}}
        ],
    )
    result = agent.process_question(
        QUESTION,
        pdf_dir=str(pdf_dir),
        index_dir=str(tmp_path / "index"),
        search_papers=False,
        frozen_chemistry_context=_context(),
        receipt_sink=receipts.append,
    )

    assert result["answer"] == "固定文献回答"
    assert result["literature_review"] == "固定文献回答"
    assert receipts
    assert all("response_contract" not in receipt["request"] for receipt in receipts)


def test_retrieval_public_signatures_do_not_expose_recorder() -> None:
    """记录器通过既有 receipt_sink 接入，不污染 Retrieval 业务签名。"""

    for name in (
        "process_question",
        "_resolve_chemistry_context",
        "extract_chemistry_context",
        "extract_keywords",
        "answer_question",
        "generate_literature_review",
        "_call_retrieval_provider",
    ):
        assert "model_output_recorder" not in inspect.signature(
            getattr(RetrievalAgent, name)
        ).parameters
