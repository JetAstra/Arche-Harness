from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest

from chemistry_multiagent.agents import retrieval_agent as retrieval_module
from tests.fakes.retrieval import make_test_retrieval_agent


def _write_pdf(path: Path, text: str = "fixture") -> bytes:
    """写入一个可读单页 PDF，并返回其字节内容。"""

    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    content = document.tobytes()
    document.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return content


def _write_v1_manifest(path: Path, pdf_path: Path, content: bytes) -> None:
    """写入 no-search 兼容的 v1 冻结 manifest。"""

    import hashlib

    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "completed",
                "target_papers": 1,
                "actual_papers": 1,
                "target_reached": True,
                "accepted_papers": [
                    {
                        "path": str(pdf_path),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_explicit_v1_no_search_manifest_is_fail_closed_without_network_or_hypothesis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """v1 冻结语料可复用；缺失或篡改 manifest/PDF 都在索引前失败。"""

    pdf_dir = tmp_path / "papers"
    pdf_path = pdf_dir / "paper.pdf"
    content = _write_pdf(pdf_path)
    manifest = tmp_path / "manifest.json"
    _write_v1_manifest(manifest, pdf_path, content)
    agent = make_test_retrieval_agent(deepseek_api_key="")
    agent.documents = []
    monkeypatch.setattr(agent, "load_index", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(agent, "build_index", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(agent, "extract_chemistry_context", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(agent, "answer_question", lambda *_args, **_kwargs: "answer")
    monkeypatch.setattr(agent, "generate_literature_review", lambda *_args, **_kwargs: "review")
    monkeypatch.setattr(agent, "extract_keywords", lambda *_args, **_kwargs: ["fixture"])
    monkeypatch.setattr(agent, "answer_question", lambda *_args, **_kwargs: "answer")
    monkeypatch.setattr(agent, "generate_literature_review", lambda *_args, **_kwargs: "review")
    monkeypatch.setattr(agent, "search_papers", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("no-search 不得联网")))
    monkeypatch.setattr(retrieval_module, "_paper_search_http_request", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("不得联网")))

    accepted = agent.process_question(
        "复用冻结论文。",
        pdf_dir=str(pdf_dir),
        index_dir=str(tmp_path / "index"),
        search_papers=False,
        paper_search={"manifest_path": str(manifest)},
    )
    assert "error" not in accepted

    missing = agent.process_question(
        "缺失 manifest。",
        pdf_dir=str(pdf_dir),
        index_dir=str(tmp_path / "missing-index"),
        search_papers=False,
        paper_search={"manifest_path": str(tmp_path / "missing.json")},
    )
    assert "error" in missing
    pdf_path.write_bytes(content + b"tampered")
    tampered = agent.process_question(
        "篡改 PDF。",
        pdf_dir=str(pdf_dir),
        index_dir=str(tmp_path / "tampered-index"),
        search_papers=False,
        paper_search={"manifest_path": str(manifest)},
    )
    assert "error" in tampered


def test_default_no_search_manifest_is_required_without_search_or_index_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """未显式传 manifest 时也只复用默认冻结文件，缺失或篡改必须先失败。"""

    pdf_dir = tmp_path / "papers"
    pdf_path = pdf_dir / "paper.pdf"
    content = _write_pdf(pdf_path)
    manifest = pdf_dir / ".paper-search-manifest.json"
    _write_v1_manifest(manifest, pdf_path, content)
    agent = make_test_retrieval_agent(deepseek_api_key="")
    agent.documents = []
    monkeypatch.setattr(agent, "load_index", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(agent, "build_index", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(agent, "extract_chemistry_context", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(agent, "answer_question", lambda *_args, **_kwargs: "answer")
    monkeypatch.setattr(agent, "generate_literature_review", lambda *_args, **_kwargs: "review")
    monkeypatch.setattr(
        agent,
        "search_papers",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("no-search 不得调用搜索")
        ),
    )
    monkeypatch.setattr(
        retrieval_module,
        "_paper_search_http_request",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("no-search 不得访问网络")
        ),
    )

    accepted = agent.process_question(
        "复用默认冻结论文。",
        pdf_dir=str(pdf_dir),
        index_dir=str(tmp_path / "index"),
        search_papers=False,
    )
    assert "error" not in accepted

    manifest.unlink()
    missing = agent.process_question(
        "默认 manifest 缺失。",
        pdf_dir=str(pdf_dir),
        index_dir=str(tmp_path / "missing-index"),
        search_papers=False,
    )
    assert "error" in missing

    _write_v1_manifest(manifest, pdf_path, content)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["target_reached"] = False
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    tampered = agent.process_question(
        "默认 manifest 被篡改。",
        pdf_dir=str(pdf_dir),
        index_dir=str(tmp_path / "tampered-index"),
        search_papers=False,
    )
    assert "error" in tampered


def test_index_marker_corpus_model_identity_and_duplicate_sha_guards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """索引只在提交标记、语料和模型身份全部不变时复用，重复 SHA 只索引一次。"""

    pdf_dir = tmp_path / "papers"
    content = _write_pdf(pdf_dir / "one.pdf", "same")
    (pdf_dir / "duplicate.pdf").write_bytes(content)
    index_dir = tmp_path / "index"
    agent = make_test_retrieval_agent(deepseek_api_key="")
    monkeypatch.setattr(agent, "_load_pdf_text", lambda _path: [("same text", 1)])
    monkeypatch.setattr(agent, "_chunk_text", lambda text: [text])
    agent.build_index(str(pdf_dir), str(index_dir))
    assert len(agent.documents) == 1

    reused = make_test_retrieval_agent(deepseek_api_key="")
    assert reused.load_index(str(index_dir), pdf_dir=str(pdf_dir)) is True
    marker = index_dir / "index.commit.json"
    marker.write_text("{", encoding="utf-8")
    assert reused.load_index(str(index_dir), pdf_dir=str(pdf_dir)) is False
    assert reused.index is None and reused.documents == []

    agent.build_index(str(pdf_dir), str(index_dir))
    (pdf_dir / "changed.pdf").write_bytes(_write_pdf(tmp_path / "changed-source.pdf", "changed"))
    assert reused.load_index(str(index_dir), pdf_dir=str(pdf_dir)) is False
    assert reused.index is None and reused.documents == []
    (pdf_dir / "changed.pdf").unlink()
    agent.build_index(str(pdf_dir), str(index_dir))
    monkeypatch.setattr(reused, "_embedding_model_identity", lambda: {"fingerprint": "changed"})
    assert reused.load_index(str(index_dir), pdf_dir=str(pdf_dir)) is False
    assert reused.index is None and reused.documents == []


def test_formal_search_reuses_partial_pdf_fairly_and_skips_provider_at_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """已有 PDF 计入缺口，配额动态重算；达到目标时不触发 provider。"""

    pdf_dir = tmp_path / "papers"
    _write_pdf(pdf_dir / "existing.pdf")
    agent = make_test_retrieval_agent(deepseek_api_key="")
    targets: list[int] = []

    def fake_single(**kwargs):
        targets.append(kwargs["keyword_target_papers"])
        known_hashes = kwargs["known_hashes"]
        while len(known_hashes) < kwargs["keyword_target_papers"]:
            digest = f"{len(known_hashes) + 1:064d}"
            known_hashes[digest] = str(pdf_dir / f"generated-{len(known_hashes)}.pdf")
        return {"keyword": kwargs["keyword"], "status": "success", "downloaded_files": []}

    monkeypatch.setattr(agent, "_search_single_keyword_with_provider_strategy", fake_single)
    result = agent.search_papers(
        ["one", "two", "three"], str(pdf_dir), provider_strategy="pypaperbot_scholar",
        target_papers=4, keyword_delay_seconds=0, keyword_jitter_seconds=0,
        download_delay_seconds=0,
    )
    assert targets == [2, 3, 4]
    assert len(result) == 4

    capped = make_test_retrieval_agent(deepseek_api_key="")
    monkeypatch.setattr(retrieval_module, "_paper_search_http_request", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("达到 cap 不得请求 provider")))
    assert capped.search_papers(["one"], str(pdf_dir), provider_strategy="pypaperbot_scholar", target_papers=1, keyword_delay_seconds=0, download_delay_seconds=0)
