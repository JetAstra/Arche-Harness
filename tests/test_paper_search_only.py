import json

import pytest

from chemistry_multiagent.agents import retrieval_agent as retrieval_module
from chemistry_multiagent.agents.retrieval_agent import (
    PaperSearchTimeoutError,
    allocate_fair_paper_search_limits,
)
from tests.fakes.retrieval import make_test_retrieval_agent

# 先导入 Retrieval，让 SentenceTransformer 先锁定标准环境的 C++ 运行库，再加载 fitz。
import fitz


CASE1_QUESTION = (
    "I plan to verify the mechanism of an asymmetric catalytic reaction using the reaction of "
    "p-nitrobenzaldehyde O=Cc1ccc(cc1)[N+](=O)[O-] with CC(=O)C under the catalysis of "
    "C(F)(F)(F)S(=O)(=O)O and N1CCC[C@H]1C(N2CCCC2), at 30°C in acetone solvent, to obtain "
    "the product CC(=O)C(O)c1ccc([N+](=O)[O-])cc1. Could you provide a specific approach "
    "to validate the reaction mechanism?\n\n"
    "Experimentally reported enantiomeric excess: 88%."
)


def test_paper_search_only_writes_manifest_and_stops_before_index(tmp_path, monkeypatch):
    agent = make_test_retrieval_agent(deepseek_api_key="")
    keywords = [
        "asymmetric catalysis",
        "aldol reactivity",
        "enantioselectivity",
        "reaction mechanism",
        "density functional theory",
    ]
    observed = {}

    monkeypatch.setattr(agent, "extract_keywords", lambda question, **kwargs: keywords)

    def fake_search(
        search_keywords,
        pdf_dir,
        status_sink,
        **kwargs,
    ):
        observed["search_keywords"] = search_keywords
        observed["candidate_page_size"] = kwargs["candidate_page_size"]
        agent._paper_search_audit["keyword_allocations"] = {
            keyword: 10 for keyword in search_keywords
        }
        agent._paper_search_audit["keyword_allocations_applied"] = True
        pdf_path = tmp_path / pdf_dir / "paper.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        document = fitz.open()
        document.new_page()
        document.save(str(pdf_path))
        document.close()
        status_sink(
            {
                "keyword": search_keywords[0],
                "status": "success",
                "downloaded_files": [str(pdf_path)],
            }
        )
        return [str(pdf_path)]

    monkeypatch.setattr(agent, "search_papers", fake_search)
    for stage in ("build_index", "answer_question", "generate_literature_review"):
        monkeypatch.setattr(
            agent,
            stage,
            lambda *args, _stage=stage, **kwargs: (_ for _ in ()).throw(
                AssertionError(f"PaperSearch-only 不得调用 { _stage }")
            ),
        )

    manifest_path = tmp_path / "manifest.json"
    manifest = agent.run_paper_search_only(
        question=CASE1_QUESTION,
        pdf_dir=str(tmp_path / "papers"),
        manifest_path=str(manifest_path),
        candidate_page_size=7,
    )

    assert observed == {
        "search_keywords": keywords,
        "candidate_page_size": 7,
    }
    assert manifest["question"] == CASE1_QUESTION
    assert manifest["keywords"] == keywords
    assert manifest["stopped_after"] == "paper_search"
    assert manifest["index_built"] is False
    assert manifest["downstream_stages_run"] == []
    assert manifest["downloaded_pdf_count"] == 1
    assert manifest["search_config"]["candidate_page_size"] == 7
    assert "limit_per_keyword" not in manifest["search_config"]
    assert manifest["search_config"]["keyword_allocations_applied"] is True
    assert manifest["search_config"]["keyword_allocations"] == {
        keyword: 10 for keyword in keywords
    }
    assert manifest["search_config"]["allocation_strategy"] == (
        "dynamic_remaining_gap_fair_allocation"
    )
    assert manifest["target_reached"] is False
    assert manifest["status"] == "partial"
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest


def test_paper_search_only_records_keyword_failures(tmp_path, monkeypatch):
    agent = make_test_retrieval_agent(deepseek_api_key="")
    monkeypatch.setattr(
        agent,
        "extract_keywords",
        lambda question, **kwargs: [
            "failed query",
            "aldol reaction",
            "enantioselectivity",
            "organocatalysis",
            "transition state",
        ],
    )

    def fake_search(
        search_keywords,
        pdf_dir,
        status_sink,
        **kwargs,
    ):
        status_sink(
            {
                "keyword": search_keywords[0],
                "status": "failed",
                "downloaded_files": [],
                "error": "network timeout",
            }
        )
        return []

    monkeypatch.setattr(agent, "search_papers", fake_search)
    manifest = agent.run_paper_search_only(
        question=CASE1_QUESTION,
        pdf_dir=str(tmp_path / "papers"),
        manifest_path=str(tmp_path / "manifest.json"),
    )

    assert manifest["downloaded_pdf_count"] == 0
    assert manifest["keyword_results"][0]["status"] == "failed"
    assert manifest["keyword_results"][0]["error"] == "network timeout"


def test_global_target_is_allocated_fairly_in_keyword_order():
    assert allocate_fair_paper_search_limits(
        ["a", "b", "c", "d"],
        10,
    ) == {"a": 3, "b": 3, "c": 2, "d": 2}


def test_search_papers_separates_global_target_from_candidate_page_size(tmp_path, monkeypatch):
    calls = []

    class FakePaperScraper:
        lib = None

        @staticmethod
        def search_papers(keyword, **kwargs):
            calls.append((keyword, kwargs))
            return {}

    monkeypatch.setattr(retrieval_module, "paperscraper", FakePaperScraper())
    monkeypatch.setattr(retrieval_module, "PAPERSCRAPER_AVAILABLE", True)
    agent = make_test_retrieval_agent(deepseek_api_key="")

    agent.search_papers(
        ["a", "b"],
        str(tmp_path / "papers"),
        target_papers=5,
        candidate_page_size=17,
        provider_strategy="paper_scraper",
        keyword_delay_seconds=0,
    )

    assert [item[1]["limit"] for item in calls] == [3, 2]
    assert [item[1]["_limit"] for item in calls] == [17, 17]


def test_candidate_page_size_must_be_positive_integer(tmp_path, monkeypatch):
    class FakePaperScraper:
        lib = None

    monkeypatch.setattr(retrieval_module, "paperscraper", FakePaperScraper())
    monkeypatch.setattr(retrieval_module, "PAPERSCRAPER_AVAILABLE", True)
    agent = make_test_retrieval_agent(deepseek_api_key="")

    with pytest.raises(ValueError, match="候选页大小"):
        agent.search_papers(
            ["a"],
            str(tmp_path / "papers"),
            candidate_page_size=0,
            provider_strategy="paper_scraper",
        )


def test_paper_search_records_real_timeout_and_continues(tmp_path, monkeypatch):
    class FakePaperScraper:
        lib = None

        @staticmethod
        def search_papers(*args, **kwargs):
            raise PaperSearchTimeoutError("模拟超时")

    monkeypatch.setattr(retrieval_module, "paperscraper", FakePaperScraper())
    monkeypatch.setattr(retrieval_module, "PAPERSCRAPER_AVAILABLE", True)
    statuses = []
    agent = make_test_retrieval_agent(deepseek_api_key="")
    paths = agent.search_papers(
        ["a", "b"],
        str(tmp_path / "papers"),
        target_papers=4,
        timeout_seconds=1,
        keyword_delay_seconds=0,
        status_sink=statuses.append,
    )

    assert paths == []
    assert [item["status"] for item in statuses] == ["timeout", "timeout"]
    assert all(item["timeout_seconds"] == 1 for item in statuses)


def test_first_semantic_scholar_429_switches_to_openalex_and_stops_at_target(
    tmp_path, monkeypatch
):
    """首个 S2 429 后应只切到 OpenAlex，并以可读 PDF 去重后停止。"""

    agent = make_test_retrieval_agent(deepseek_api_key="")
    keywords = [
        "asymmetric aldol reaction",
        "p-nitrobenzaldehyde",
        "acetone",
        "organocatalysis",
        "transition state analysis",
    ]
    monkeypatch.setattr(agent, "extract_keywords", lambda question, **kwargs: keywords)

    pdf_fixtures = {}
    for index in range(51):
        document = fitz.open()
        page = document.new_page()
        page.insert_text((72, 72), f"Open access result {index}")
        pdf_fixtures[index] = document.tobytes()
        document.close()
    openalex_candidates = []
    for index in range(51):
        openalex_candidates.append(
            {
                "id": f"https://openalex.org/W{index}",
                "doi": f"https://doi.org/10.1234/example-{index}",
                "title": f"Open access result {index}",
                "type": "article",
                "is_retracted": False,
                "open_access": {"is_oa": True},
                "best_oa_location": {
                    "pdf_url": "https://example.test/repeated.pdf"
                    if index == 50
                    else f"https://example.test/{index}.pdf"
                },
                "locations": [],
            }
        )
    calls = []

    def fake_http_request(url, **kwargs):
        calls.append((url, kwargs))
        if "semanticscholar.org" in url:
            return {
                "status": 429,
                "headers": {"Retry-After": "120"},
                "body": b"rate limited",
            }
        if "api.openalex.org" in url:
            return {
                "status": 200,
                "headers": {"x-ratelimit-remaining": "999"},
                "json": {"meta": {"count": 51}, "results": openalex_candidates},
            }
        if url.endswith("repeated.pdf"):
            return {"status": 200, "headers": {}, "body": pdf_fixtures[0]}
        index = int(url.rsplit("/", 1)[-1].split(".", 1)[0])
        return {"status": 200, "headers": {}, "body": pdf_fixtures[index]}

    monkeypatch.setattr(retrieval_module, "_paper_search_http_request", fake_http_request)

    manifest = agent.run_paper_search_only(
        question=CASE1_QUESTION,
        pdf_dir=str(tmp_path / "papers"),
        manifest_path=str(tmp_path / "manifest.json"),
        target_papers=50,
        provider_strategy="semantic_scholar_then_openalex",
        keyword_delay_seconds=0,
        download_delay_seconds=0,
    )

    assert manifest["stopped_after"] == "paper_search"
    assert manifest["target_reached"] is True
    assert manifest["downloaded_pdf_count"] == 50
    assert len(manifest["accepted_papers"]) == 50
    assert all(
        set(("path", "sha256", "provider", "keyword", "identifiers", "source_host"))
        <= set(item)
        for item in manifest["accepted_papers"]
    )
    assert manifest["search_config"]["provider_strategy"] == "semantic_scholar_then_openalex"
    providers = manifest["provider_attempts"]
    assert providers[0]["provider"] == "semantic_scholar"
    assert providers[0]["http_statuses"] == [429]
    assert providers[0]["retry_count"] == 0
    assert any(item["provider"] == "openalex" for item in providers)
    assert len(list((tmp_path / "papers").glob("*.pdf"))) == 50
    assert all(path.read_bytes().startswith(b"%PDF") for path in (tmp_path / "papers").glob("*.pdf"))
    assert not any(path.name.endswith(".part") for path in (tmp_path / "papers").iterdir())
    assert not any("secret-test-key" in json.dumps(item) for item in providers)


def test_google_scholar_html_parser_extracts_public_pdf_candidates():
    html = """
    <html><body>
      <div class="gs_r gs_or gs_scl" data-cid="cid-1">
        <div class="gs_ggs"><div class="gs_ggsd">
          <div class="gs_or_ggsm"><a href="https://repository.test/one.pdf">[PDF]</a></div>
        </div></div>
        <div class="gs_ri">
          <h3 class="gs_rt"><a href="https://publisher.test/one">First study</a></h3>
          <div class="gs_a">A. Author - Journal of Chemistry - 2024</div>
          <div class="gs_rs">A useful abstract about benzaldehyde.</div>
        </div>
      </div>
      <div class="gs_r gs_or gs_scl" data-cid="cid-2">
        <div class="gs_ri">
          <h3 class="gs_rt"><a href="https://publisher.test/two">Second study</a></h3>
          <div class="gs_a">B. Author - 2023</div>
          <div class="gs_rs">No public copy is exposed here.</div>
        </div>
      </div>
    </body></html>
    """

    candidates = retrieval_module._parse_google_scholar_html(html)

    assert candidates == [
        {
            "scholar_id": "cid-1",
            "title": "First study",
            "authors": "A. Author - Journal of Chemistry - 2024",
            "snippet": "A useful abstract about benzaldehyde.",
            "result_url": "https://publisher.test/one",
            "scholar_pdf_url": "https://repository.test/one.pdf",
        },
        {
            "scholar_id": "cid-2",
            "title": "Second study",
            "authors": "B. Author - 2023",
            "snippet": "No public copy is exposed here.",
            "result_url": "https://publisher.test/two",
        },
    ]


def test_google_scholar_html_parser_handles_live_void_elements():
    """真实 Scholar 页面中的未闭合 void 元素不得破坏结果容器深度。"""

    html = """
    <!doctype html><html><head>
      <meta http-equiv="Content-Type" content="text/html;charset=UTF-8">
      <link rel="stylesheet" href="/scholar.css">
    </head><body>
      <form><input name="q"><input type="hidden" name="hl" value="en">
        <div class="gs_r gs_or gs_scl" data-cid="live-cid">
          <div class="gs_ri">
            <h3 class="gs_rt"><a href="https://publisher.test/live">Live result</a></h3>
            <div class="gs_a">A. Author - 2024</div>
            <div class="gs_rs">A live-shaped Scholar result.</div>
            <div class="gs_ggs"><div class="gs_ggsd">
              <div class="gs_or_ggsm"><a href="https://repository.test/live.pdf">[PDF]</a></div>
            </div></div>
          </div>
        </div>
      </form>
    </body></html>
    """

    candidates = retrieval_module._parse_google_scholar_html(html)

    assert candidates == [
        {
            "scholar_id": "live-cid",
            "title": "Live result",
            "authors": "A. Author - 2024",
            "snippet": "A live-shaped Scholar result.",
            "result_url": "https://publisher.test/live",
            "scholar_pdf_url": "https://repository.test/live.pdf",
        }
    ]


def test_pypaperbot_scholar_captcha_is_structured_failure_without_waiting(
    tmp_path, monkeypatch
):
    agent = make_test_retrieval_agent(deepseek_api_key="")
    calls = []

    def fake_http_request(url, **kwargs):
        calls.append((url, kwargs))
        return {
            "status": 200,
            "headers": {},
            "body": b"<html><title>Google Scholar</title>unusual traffic CAPTCHA</html>",
        }

    monkeypatch.setattr(retrieval_module, "_paper_search_http_request", fake_http_request)
    sleeps = []
    monkeypatch.setattr(agent, "_sleep_paper_search_with_backoff", sleeps.append)

    manifest = agent.run_paper_search_only(
        question=CASE1_QUESTION,
        pdf_dir=str(tmp_path / "papers"),
        manifest_path=str(tmp_path / "manifest.json"),
        keywords_override=["one", "two", "three", "four", "five"],
        provider_strategy="pypaperbot_scholar",
        scholar_max_pages=3,
        scholar_results_per_page=10,
        scholar_crossref_max_candidates=0,
        keyword_delay_seconds=0,
        download_delay_seconds=0,
    )

    assert len(calls) == 1
    assert sleeps == []
    attempt = manifest["provider_attempts"][0]
    assert attempt["provider"] == "pypaperbot_scholar"
    assert attempt["structured_failure"] == "blocked"
    assert attempt["blocked_reason"] == "captcha_or_unusual_traffic"
    assert manifest["downloaded_pdf_count"] == 0


def test_pypaperbot_scholar_downloads_only_exposed_public_pdf_and_rejects_shadow_url(
    tmp_path, monkeypatch
):
    agent = make_test_retrieval_agent(deepseek_api_key="")
    document = fitz.open()
    document.new_page().insert_text((72, 72), "public fixture")
    pdf_bytes = document.tobytes()
    document.close()
    calls = []

    scholar_html = """
    <div class="gs_r gs_or gs_scl" data-cid="public">
      <div class="gs_ggs"><div class="gs_or_ggsm">
        <a href="https://repository.test/public.pdf">[PDF]</a>
        <a href="https://sci-hub.se/secret.pdf">[PDF]</a>
      </div></div>
      <div class="gs_ri">
        <h3 class="gs_rt"><a href="https://publisher.test/public">Public</a></h3>
        <div class="gs_a">Author - 2024</div>
        <div class="gs_rs">Public abstract.</div>
      </div>
    </div>
    """

    def fake_http_request(url, **kwargs):
        calls.append(url)
        if "scholar.google.com" in url:
            return {"status": 200, "headers": {}, "body": scholar_html.encode()}
        assert url == "https://repository.test/public.pdf"
        return {"status": 200, "headers": {"content-type": "application/pdf"}, "body": pdf_bytes}

    monkeypatch.setattr(retrieval_module, "_paper_search_http_request", fake_http_request)
    manifest = agent.run_paper_search_only(
        question=CASE1_QUESTION,
        pdf_dir=str(tmp_path / "papers"),
        manifest_path=str(tmp_path / "manifest.json"),
        keywords_override=["one", "two", "three", "four", "five"],
        provider_strategy="pypaperbot_scholar",
        scholar_max_pages=1,
        scholar_results_per_page=10,
        scholar_crossref_max_candidates=0,
        target_papers=1,
        keyword_delay_seconds=0,
        download_delay_seconds=0,
    )

    assert calls == ["https://scholar.google.com/scholar", "https://repository.test/public.pdf"]
    assert manifest["downloaded_pdf_count"] == 1
    assert "sci-hub" not in json.dumps(manifest)
    assert manifest["accepted_papers"][0]["provider"] == "pypaperbot_scholar"


def test_pypaperbot_scholar_uses_crossref_only_for_matching_metadata(
    tmp_path, monkeypatch
):
    agent = make_test_retrieval_agent(deepseek_api_key="")
    document = fitz.open()
    document.new_page().insert_text((72, 72), "Crossref fixture")
    pdf_bytes = document.tobytes()
    document.close()
    calls = []
    scholar_html = """
    <div class="gs_r gs_or gs_scl" data-cid="crossref-id">
      <div class="gs_ggs"><a href="https://repository.test/crossref.pdf">[PDF]</a></div>
      <div class="gs_ri"><h3 class="gs_rt"><a>Crossref matching title</a></h3></div>
    </div>
    """

    def fake_http_request(url, **kwargs):
        calls.append((url, kwargs))
        if "scholar.google.com" in url:
            return {"status": 200, "headers": {}, "body": scholar_html.encode()}
        if "api.crossref.org" in url:
            return {
                "status": 200,
                "headers": {},
                "json": {
                    "message": {
                        "items": [
                            {
                                "DOI": "10.1234/crossref-fixture",
                                "title": ["Crossref matching title"],
                                "short-container-title": ["Fixture Journal"],
                            }
                        ]
                    }
                },
            }
        assert url == "https://repository.test/crossref.pdf"
        return {"status": 200, "headers": {}, "body": pdf_bytes}

    monkeypatch.setattr(retrieval_module, "_paper_search_http_request", fake_http_request)
    manifest = agent.run_paper_search_only(
        question=CASE1_QUESTION,
        pdf_dir=str(tmp_path / "papers"),
        manifest_path=str(tmp_path / "manifest.json"),
        keywords_override=["one", "two", "three", "four", "five"],
        provider_strategy="pypaperbot_scholar",
        scholar_max_pages=1,
        scholar_results_per_page=10,
        scholar_crossref_max_candidates=1,
        target_papers=1,
        keyword_delay_seconds=0,
        download_delay_seconds=0,
    )

    assert [url for url, _ in calls] == [
        "https://scholar.google.com/scholar",
        "https://api.crossref.org/works",
        "https://repository.test/crossref.pdf",
    ]
    assert manifest["accepted_papers"][0]["identifiers"] == [
        "doi:10.1234/crossref-fixture",
        "google_scholar:crossref-id",
    ]
    assert manifest["provider_attempts"][0]["crossref_matched_count"] == 1


def test_pypaperbot_scholar_deduplicates_keywords_and_stops_at_global_target(
    tmp_path, monkeypatch
):
    agent = make_test_retrieval_agent(deepseek_api_key="")
    documents = []
    for label in ("same", "unique"):
        document = fitz.open()
        document.new_page().insert_text((72, 72), label)
        documents.append(document.tobytes())
        document.close()
    calls = []
    candidate_html = {
        "one": """
        <div class="gs_r gs_or gs_scl" data-cid="same-id">
          <div class="gs_ggs"><a href="https://repository.test/same.pdf">[PDF]</a></div>
          <div class="gs_ri"><h3 class="gs_rt"><a>Same</a></h3></div>
        </div>
        <div class="gs_r gs_or gs_scl" data-cid="unique-id">
          <div class="gs_ggs"><a href="https://repository.test/unique.pdf">[PDF]</a></div>
          <div class="gs_ri"><h3 class="gs_rt"><a>Unique</a></h3></div>
        </div>
        """,
            "two": """
        <div class="gs_r gs_or gs_scl" data-cid="same-id">
          <div class="gs_ggs"><a href="https://repository.test/same-again.pdf">[PDF]</a></div>
          <div class="gs_ri"><h3 class="gs_rt"><a>Same again</a></h3></div>
            </div>
            """,
            "three": """
            <div class="gs_r gs_or gs_scl" data-cid="third-id">
              <div class="gs_ggs"><a href="https://repository.test/third.pdf">[PDF]</a></div>
              <div class="gs_ri"><h3 class="gs_rt"><a>Third</a></h3></div>
            </div>
            """,
    }

    def fake_http_request(url, **kwargs):
        calls.append((url, kwargs))
        if "scholar.google.com" in url:
            query = kwargs["params"]["q"]
            return {"status": 200, "headers": {}, "body": candidate_html.get(query, "").encode()}
        if url.endswith("same.pdf") or url.endswith("same-again.pdf"):
            return {"status": 200, "headers": {}, "body": documents[0]}
        if url.endswith("third.pdf"):
            document = fitz.open()
            document.new_page().insert_text((72, 72), "third")
            body = document.tobytes()
            document.close()
            return {"status": 200, "headers": {}, "body": body}
        return {"status": 200, "headers": {}, "body": documents[1]}

    monkeypatch.setattr(retrieval_module, "_paper_search_http_request", fake_http_request)
    manifest = agent.run_paper_search_only(
        question=CASE1_QUESTION,
        pdf_dir=str(tmp_path / "papers"),
        manifest_path=str(tmp_path / "manifest.json"),
        keywords_override=["one", "two", "three", "four", "five"],
        provider_strategy="pypaperbot_scholar",
        scholar_max_pages=2,
        scholar_results_per_page=10,
        scholar_crossref_max_candidates=0,
        target_papers=2,
        keyword_delay_seconds=0,
        download_delay_seconds=0,
    )

    scholar_calls = [item for item in calls if "scholar.google.com" in item[0]]
    assert [item[1]["params"]["q"] for item in scholar_calls] == [
        "one",
        "two",
        "three",
    ]
    allocations = manifest["search_config"]["keyword_allocations"]
    assert allocations["one"] == 1
    assert allocations["two"] == 1
    assert allocations["three"] == 1
    actual_added = {
        item["keyword"]: item["actual_added"]
        for item in manifest["keyword_results"]
    }
    assert actual_added == {"one": 1, "two": 0, "three": 1}
    assert manifest["downloaded_pdf_count"] == 2
    assert manifest["target_reached"] is True
    assert manifest["statistics"]["target_stop_reached"] is True


def test_cli_defaults_to_pypaperbot_scholar_provider():
    from scripts.run_paper_search_only import build_paper_search_only_parser

    args = build_paper_search_only_parser().parse_args(["--question", "question"])

    assert args.provider_strategy == "pypaperbot_scholar"
    assert args.scholar_max_pages == 3
    assert args.scholar_results_per_page == 10
    assert args.scholar_crossref_max_candidates == 3
    assert args.pdf_dir == "local_data/papers/case1-paper-search/papers"
    assert args.manifest == "local_data/papers/case1-paper-search/manifest.json"


def test_formal_scholar_dynamic_allocation_caps_each_keyword_and_rebalances(
    tmp_path, monkeypatch
):
    """正式 Scholar 每轮只消费公平配额，前一关键词失败后缺口由后续关键词承接。"""

    agent = make_test_retrieval_agent(deepseek_api_key="")
    keywords = ["one", "two", "three"]
    allocations = []

    def fake_single(**kwargs):
        target = kwargs["keyword_target_papers"]
        allocations.append(target)
        known_hashes = kwargs["known_hashes"]
        if kwargs["keyword"] != "one":
            while len(known_hashes) < target:
                digest = f"{len(known_hashes) + 1:064d}"
                known_hashes[digest] = str(tmp_path / "papers" / f"{digest}.pdf")
        return {
            "keyword": kwargs["keyword"],
            "status": "success" if kwargs["keyword"] != "one" else "no_valid_pdf",
            "downloaded_files": [],
            "provider": "pypaperbot_scholar",
            "target_count": len(known_hashes),
        }

    monkeypatch.setattr(
        agent,
        "_search_single_keyword_with_provider_strategy",
        fake_single,
    )
    result = agent.search_papers(
        keywords,
        str(tmp_path / "papers"),
        provider_strategy="pypaperbot_scholar",
        target_papers=4,
        keyword_delay_seconds=0,
        keyword_jitter_seconds=0,
        scholar_request_delay_seconds=0,
        scholar_request_jitter_seconds=0,
        scholar_initial_delay_seconds=0,
        scholar_initial_jitter_seconds=0,
        scholar_backoff_base_seconds=0,
        scholar_max_backoff_seconds=0,
        download_delay_seconds=0,
    )
    assert allocations == [2, 2, 4]
    assert len(result) == 4
