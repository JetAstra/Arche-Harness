import json
from pathlib import Path

import numpy as np
import pytest

from chemistry_multiagent.agents import retrieval_agent as retrieval_module
from chemistry_multiagent.agents.retrieval_agent import RetrievalEmbeddingError
from chemistry_multiagent.controllers import chemistry_multiagent_controller as controller_module
import faiss
import fitz
from tests.fakes.retrieval import FakeSentenceTransformer, make_test_retrieval_agent


def test_no_search_still_selects_bge_embedder():
    assert controller_module._select_retrieval_embedder_name(search_papers=False) == "bge"
    assert controller_module._select_retrieval_embedder_name(search_papers=True) == "bge"


def test_controller_passes_bge_retrieval_embedder_name(monkeypatch, tmp_path):
    seen = {}

    class FakeRetrievalAgent:
        def __init__(self, deepseek_api_key=None, embedder_name="bge"):
            seen["embedder_name"] = embedder_name

    monkeypatch.setattr(controller_module, "RetrievalAgent", FakeRetrievalAgent)
    monkeypatch.setattr(controller_module, "HypothesisAgent", lambda **kwargs: object())

    controller_module.ResearchTaskApplication(
        work_dir=str(tmp_path),
        memory_enabled=False,
        retrieval_embedder_name="bge",
    )

    assert seen["embedder_name"] == "bge"


def test_injected_encoder_skips_sentence_transformer_loading(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("注入测试编码器时不应再次加载 SentenceTransformer")

    monkeypatch.setattr(retrieval_module, "SentenceTransformer", fail_if_called)
    agent = make_test_retrieval_agent(deepseek_api_key="")

    assert agent.embedder_name == "bge"
    assert agent._effective_embedder_name == "bge"
    assert agent._get_embeddings(["semantic retrieval"]).shape == (1, 8)


def test_simple_embedder_is_rejected():
    with pytest.raises(ValueError, match="只支持 SentenceTransformer"):
        retrieval_module.RetrievalAgent(deepseek_api_key="", embedder_name="simple")


def test_missing_sentence_transformer_dependency_fails_fast(monkeypatch, tmp_path):
    monkeypatch.setattr(retrieval_module, "SENTENCE_TRANSFORMERS_AVAILABLE", False)
    monkeypatch.setattr(retrieval_module, "SentenceTransformer", None)

    with pytest.raises(RetrievalEmbeddingError, match="sentence-transformers"):
        retrieval_module.RetrievalAgent(
            deepseek_api_key="",
            model_path=str(tmp_path / "bge-model"),
        )


def test_missing_local_model_directory_fails_fast(tmp_path):
    with pytest.raises(RetrievalEmbeddingError, match="目录不存在"):
        retrieval_module.RetrievalAgent(
            deepseek_api_key="",
            model_path=str(tmp_path / "missing-bge-model"),
        )


def test_bge_uses_explicit_local_model_path_and_cpu(monkeypatch, tmp_path):
    model_path = tmp_path / "bge-model"
    model_path.mkdir()
    seen = {}

    class FakeSentenceTransformerLoader:
        def __init__(self, model_name, **kwargs):
            seen["model_name"] = model_name
            seen.update(kwargs)

        def encode(self, texts, normalize_embeddings=True):
            return np.ones((len(texts), 4), dtype="float32")

    monkeypatch.setattr(retrieval_module, "SENTENCE_TRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr(
        retrieval_module,
        "SentenceTransformer",
        FakeSentenceTransformerLoader,
    )
    monkeypatch.setenv("ARCHE_RETRIEVAL_DEVICE", "cpu")

    agent = retrieval_module.RetrievalAgent(
        deepseek_api_key="",
        model_path=str(model_path),
    )

    assert seen == {
        "model_name": str(model_path.resolve()),
        "device": "cpu",
        "local_files_only": True,
    }


def test_bge_requires_model_path_and_never_resolves_repo_id(monkeypatch):
    monkeypatch.delenv("ARCHE_RETRIEVAL_MODEL_PATH", raising=False)
    with pytest.raises(RetrievalEmbeddingError, match="禁止从模型仓库自动下载"):
        retrieval_module.RetrievalAgent(deepseek_api_key="")


def test_encode_failure_does_not_fallback_to_word_matching(tmp_path, monkeypatch):
    class FailingEncoder:
        def encode(self, texts, normalize_embeddings=True):
            raise RuntimeError("encoder exploded")

    pdf_dir = tmp_path / "papers"
    pdf_dir.mkdir()
    document = fitz.open()
    document.new_page().insert_text((72, 72), "fixture")
    document.save(str(pdf_dir / "paper.pdf"))
    document.close()
    agent = retrieval_module.RetrievalAgent(
        deepseek_api_key="",
        embedder=FailingEncoder(),
    )
    monkeypatch.setattr(agent, "_load_pdf_text", lambda _path: [("alpha beta", 1)])
    monkeypatch.setattr(agent, "_chunk_text", lambda text: [text])

    with pytest.raises(RetrievalEmbeddingError, match="语义嵌入生成失败"):
        agent.build_index(str(pdf_dir), str(tmp_path / "index"))
    assert agent._effective_embedder_name == "bge"


def test_old_simple_index_is_rejected(tmp_path):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    index = faiss.IndexFlatIP(2)
    index.add(np.ones((1, 2), dtype="float32"))
    faiss.write_index(index, str(index_dir / "index.faiss"))
    (index_dir / "docs.json").write_text(
        json.dumps(
            {
                "documents": ["legacy"],
                "metadatas": [{"source": "legacy.pdf", "page": 1}],
                "embedder_name": "simple",
                "simple_vocabulary": ["legacy", "word"],
            }
        ),
        encoding="utf-8",
    )

    agent = make_test_retrieval_agent(deepseek_api_key="")
    assert agent.load_index(str(index_dir)) is False
    assert agent.index is None


def test_no_search_accepts_missing_default_paper_directory(tmp_path, monkeypatch):
    """关闭论文搜索时，已冻结的最小语料应能通过 manifest 校验。"""

    pdf_dir = tmp_path / "papers"
    index_dir = tmp_path / "index"
    pdf_dir.mkdir(parents=True)
    document = fitz.open()
    document.new_page().insert_text((72, 72), "frozen no-search fixture")
    document.save(str(pdf_dir / "paper.pdf"))
    document.close()
    agent = make_test_retrieval_agent(deepseek_api_key="")
    agent._write_paper_search_manifest(
        question="Analyze benzaldehyde.",
        keywords=[],
        pdf_dir=str(pdf_dir),
        manifest_path=str(pdf_dir / ".paper-search-manifest.json"),
        target_papers=1,
        search_config={"target_papers": 1, "provider_strategy": "frozen_fixture"},
        keyword_results=[],
        keyword_source="override",
    )
    monkeypatch.setattr(agent, "load_index", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        agent,
        "build_index",
        lambda _pdf_dir, target_index_dir: Path(target_index_dir).mkdir(
            parents=True, exist_ok=True
        ),
    )
    monkeypatch.setattr(
        agent,
        "extract_keywords",
        lambda _question, **_kwargs: ["benzaldehyde"],
    )
    monkeypatch.setattr(
        agent,
        "extract_chemistry_context",
        lambda **_kwargs: {"target_species": "benzaldehyde"},
    )

    result = agent.process_question(
        question="Analyze benzaldehyde.",
        pdf_dir=str(pdf_dir),
        index_dir=str(index_dir),
        search_papers=False,
    )

    assert pdf_dir.is_dir()
    assert index_dir.is_dir()
    assert result["retrieved_papers"] == []
    assert "error" not in result


def test_semantic_index_reuses_injected_encoder(tmp_path, monkeypatch):
    pdf_dir = tmp_path / "papers"
    index_dir = tmp_path / "index"
    pdf_dir.mkdir()
    document = fitz.open()
    document.new_page().insert_text((72, 72), "fixture")
    document.save(str(pdf_dir / "paper.pdf"))
    document.close()

    agent = make_test_retrieval_agent(deepseek_api_key="")
    monkeypatch.setattr(agent, "_load_pdf_text", lambda _path: [("alpha beta gamma", 1)])
    monkeypatch.setattr(agent, "_chunk_text", lambda text: [text])

    agent.build_index(str(pdf_dir), str(index_dir))
    assert (index_dir / "index.faiss").is_file()
    metadata = json.loads((index_dir / "docs.json").read_text(encoding="utf-8"))
    assert metadata["embedder_name"] == "bge"
    assert metadata["embedding_backend"] == "sentence-transformers"
    assert metadata["embedding_dimension"] == 8
