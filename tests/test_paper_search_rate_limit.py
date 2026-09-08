"""PaperSearch provider 限流、回退和可恢复下载的确定性测试。"""

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
import json
import pytest

from chemistry_multiagent.agents import retrieval_agent as retrieval_module
from chemistry_multiagent.agents.retrieval_agent import (
    PaperSearchTimeoutError,
    _candidate_pdf_urls,
    _paper_search_http_request,
    _parse_retry_after_seconds,
)
from tests.fakes.retrieval import make_test_retrieval_agent

# 先导入 Retrieval，让 SentenceTransformer 先锁定标准环境的 C++ 运行库，再加载 fitz。
import fitz


def test_scholar_default_pacing_is_conservative_and_injectable():
    sleeper_calls = []

    class FixedRng:
        def uniform(self, low, high):
            return high

    agent = make_test_retrieval_agent(
        deepseek_api_key="",
        paper_search_sleeper=sleeper_calls.append,
        paper_search_rng=FixedRng(),
    )
    state = {
        "initial_delay_seconds": 10,
        "initial_jitter_seconds": 20,
        "request_delay_seconds": 15,
        "request_jitter_seconds": 15,
    }
    agent._wait_for_scholar_page_request(state)
    state["last_request_at"] = 0
    agent._paper_search_clock = lambda: 0
    agent._wait_for_scholar_page_request(state)
    assert sleeper_calls == [30, 30]


def test_malformed_keyword_delay_environment_uses_conservative_default(
    tmp_path, monkeypatch
):
    """环境变量非法时仍回退到 45 秒，而不是恢复为过快的旧默认值。"""

    agent = make_test_retrieval_agent(deepseek_api_key="")
    observed_sleeps = []
    monkeypatch.setenv("ARCHE_PAPERSEARCH_KEYWORD_DELAY_SECONDS", "invalid")
    monkeypatch.setattr(
        agent,
        "_sleep_paper_search_with_backoff",
        observed_sleeps.append,
    )
    monkeypatch.setattr(
        agent,
        "_search_single_keyword_with_provider_strategy",
        lambda **kwargs: {
            "keyword": kwargs["keyword"],
            "status": "no_valid_pdf",
            "downloaded_files": [],
            "provider": "scholar",
            "target_count": 0,
        },
    )

    agent.search_papers(
        ["one", "two"],
        str(tmp_path / "papers"),
        target_papers=1,
        provider_strategy="pypaperbot_scholar",
        keyword_delay_seconds=None,
        keyword_jitter_seconds=0,
        download_delay_seconds=0,
    )

    assert observed_sleeps == [45.0, 45.0]


def test_scholar_first_429_stops_without_short_retry(tmp_path, monkeypatch):
    agent = make_test_retrieval_agent(deepseek_api_key="")
    sleeps = []
    monkeypatch.setattr(agent, "_sleep_paper_search_with_backoff", sleeps.append)
    monkeypatch.setattr(
        retrieval_module,
        "_paper_search_http_request",
        lambda *args, **kwargs: {
            "status": 429,
            "headers": {},
            "body": b"limited",
        },
    )
    manifest = agent.run_paper_search_only(
        question="q",
        pdf_dir=str(tmp_path / "papers"),
        manifest_path=str(tmp_path / "manifest.json"),
        keywords_override=["one", "two", "three", "four", "five"],
        target_papers=1,
        keyword_delay_seconds=0,
        download_delay_seconds=0,
        scholar_max_pages=1,
        scholar_crossref_max_candidates=0,
    )
    assert manifest["provider_attempts"][0]["structured_failure"] == "rate_limited_shutdown"
    assert manifest["provider_attempts"][0]["retry_count"] == 0
    assert all(delay >= 900 for delay in manifest["provider_attempts"][0]["cooldown_seconds"])


def test_scholar_response_cache_is_reused_after_restart(tmp_path, monkeypatch):
    html = b'<div class="gs_r gs_or gs_scl" data-cid="cached"><div class="gs_ri"><h3 class="gs_rt"><a>Cached</a></h3></div></div>'
    calls = []

    def fake_request(url, **kwargs):
        calls.append(url)
        return {"status": 200, "headers": {}, "body": html}

    monkeypatch.setattr(retrieval_module, "_paper_search_http_request", fake_request)
    first = make_test_retrieval_agent(deepseek_api_key="")
    cache_dir = tmp_path / "cache"
    first.search_papers(
        ["cached"],
        str(tmp_path / "papers"),
        target_papers=1,
        provider_strategy="pypaperbot_scholar",
        scholar_max_pages=1,
        scholar_crossref_max_candidates=0,
        keyword_delay_seconds=0,
        download_delay_seconds=0,
        scholar_initial_delay_seconds=0,
        scholar_initial_jitter_seconds=0,
        response_cache_dir=str(cache_dir),
    )
    second = make_test_retrieval_agent(deepseek_api_key="")
    second.search_papers(
        ["cached"],
        str(tmp_path / "papers2"),
        target_papers=1,
        provider_strategy="pypaperbot_scholar",
        scholar_max_pages=1,
        scholar_crossref_max_candidates=0,
        keyword_delay_seconds=0,
        download_delay_seconds=0,
        scholar_initial_delay_seconds=0,
        scholar_initial_jitter_seconds=0,
        response_cache_dir=str(cache_dir),
    )
    assert calls.count("https://scholar.google.com/scholar") == 1


def test_scholar_provider_lock_rejects_concurrent_execution(tmp_path):
    agent = make_test_retrieval_agent(deepseek_api_key="")
    assert agent._paper_search_execution_lock.acquire(blocking=False)
    try:
        with pytest.raises(RuntimeError, match="并发 Scholar 请求"):
            agent.search_papers(
                ["one"],
                str(tmp_path / "papers"),
                target_papers=1,
                provider_strategy="pypaperbot_scholar",
                keyword_delay_seconds=0,
                download_delay_seconds=0,
            )
    finally:
        agent._paper_search_execution_lock.release()


def test_rate_limit_cooldown_snapshot_is_checked_before_network(tmp_path, monkeypatch):
    agent = make_test_retrieval_agent(deepseek_api_key="")
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    state_path = tmp_path / "resume.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "provider_strategy": "pypaperbot_scholar",
                "keywords": ["one", "two", "three", "four", "five"],
                "target_papers": 1,
                "corpus_files": agent._collect_pdf_corpus_records(str(papers_dir)),
                "corpus_fingerprint": agent._calculate_pdf_corpus_fingerprint(
                    str(papers_dir)
                ),
                "completed_pages": {"one": []},
                "attempted_pages": {"one": [0]},
                "completed_keywords": [],
                "next_keyword_index": 0,
                "accepted_papers": [],
                "scholar_state": {
                    "blocked": True,
                    "blocked_failure": "rate_limited_shutdown",
                    "blocked_reason": "rate_limited_http_429",
                    "cooldown_until": 4102444800,
                },
            }
        ),
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(
        retrieval_module,
        "_paper_search_http_request",
        lambda *args, **kwargs: calls.append(args) or {"status": 200, "body": b""},
    )
    agent.run_paper_search_only(
        question="q",
        pdf_dir=str(papers_dir),
        manifest_path=str(tmp_path / "manifest.json"),
        keywords_override=["one", "two", "three", "four", "five"],
        target_papers=1,
        keyword_delay_seconds=0,
        download_delay_seconds=0,
        scholar_initial_delay_seconds=0,
        scholar_initial_jitter_seconds=0,
        resume_state_path=str(state_path),
    )
    assert calls == []


def _make_pdf_bytes(label: str) -> bytes:
    document = fitz.open()
    document.new_page().insert_text((72, 72), label)
    data = document.tobytes()
    document.close()
    return data


def test_retry_after_accepts_seconds_and_http_date():
    now = datetime(2026, 8, 12, 0, 0, tzinfo=timezone.utc)
    assert _parse_retry_after_seconds({"Retry-After": "7"}, now=now) == 7
    retry_at = now + timedelta(seconds=11)
    assert _parse_retry_after_seconds(
        {"retry-after": format_datetime(retry_at, usegmt=True)}, now=now
    ) == 11


def test_pdf_candidate_urls_only_keep_http_transports():
    candidate = {
        "best_oa_location": {"pdf_url": "file:///tmp/not-remote.pdf"},
        "primary_location": {"pdf_url": "https://example.test/paper.pdf"},
        "locations": [
            {"pdf_url": "data:application/pdf;base64,unsafe"},
            {"pdf_url": "http://example.test/alternate.pdf"},
        ],
    }
    assert _candidate_pdf_urls(candidate) == [
        "https://example.test/paper.pdf",
        "http://example.test/alternate.pdf",
    ]


def test_http_reader_treats_lowercase_content_length_as_oversize(monkeypatch):
    class FakeResponse:
        status_code = 200
        headers = {"content-length": "9", "content-type": "application/json"}

        def close(self):
            return None

        def iter_content(self, **kwargs):
            raise AssertionError("已声明超大正文时不得继续读取")

    monkeypatch.setattr(
        retrieval_module.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(),
    )
    response = _paper_search_http_request(
        "https://example.test/metadata", max_bytes=8
    )
    assert response["oversize"] is True
    assert response["body"] == b""


def test_semantic_scholar_key_retries_with_backoff_and_redacts_secret(tmp_path, monkeypatch):
    agent = make_test_retrieval_agent(deepseek_api_key="")
    pdf_bytes = _make_pdf_bytes("S2 retry fixture")
    responses = [
        {"status": 429, "headers": {"Retry-After": "3"}, "body": b"limited"},
        {
            "status": 200,
            "headers": {},
            "json": {
                "data": [
                    {
                        "paperId": "s2-retry-1",
                        "title": "Retry fixture",
                        "openAccessPdf": {"url": "https://example.test/retry.pdf"},
                    }
                ]
            },
        },
    ]
    observed = []
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "secret-test-key")

    def fake_request(url, **kwargs):
        observed.append((url, kwargs))
        if url.endswith("retry.pdf"):
            return {"status": 200, "headers": {}, "body": pdf_bytes}
        return responses.pop(0)

    monkeypatch.setattr(retrieval_module, "_paper_search_http_request", fake_request)
    sleeps = []
    monkeypatch.setattr(agent, "_sleep_paper_search_with_backoff", sleeps.append)
    paths = agent.search_papers(
        ["retry query"],
        str(tmp_path / "papers"),
        target_papers=1,
        provider_strategy="semantic_scholar",
        keyword_delay_seconds=0,
        download_delay_seconds=0,
    )
    assert len(paths) == 1
    assert len([item for item in observed if "semanticscholar" in item[0]]) == 2
    assert sleeps and sleeps[0] >= 3
    attempt = agent._paper_search_audit["provider_attempts"][0]
    assert attempt["http_statuses"] == [429, 200]
    assert attempt["retry_count"] == 1
    assert "secret-test-key" not in json.dumps(agent._paper_search_audit)
    assert "x-api-key" in observed[0][1]["headers"]


def test_openalex_resume_html_fallback_sha_dedup_and_part_cleanup(tmp_path, monkeypatch):
    agent = make_test_retrieval_agent(deepseek_api_key="")
    pdfs = [_make_pdf_bytes("first"), _make_pdf_bytes("second")]
    paper_dir = tmp_path / "papers"
    paper_dir.mkdir()
    stale = paper_dir / ".stale.pdf.part"
    stale.write_bytes(b"partial")
    candidates = [
        {
            "id": "https://openalex.org/W1",
            "doi": "https://doi.org/10.1/one",
            "display_name": "First",
            "is_retracted": False,
            "open_access": {"is_oa": True},
            "best_oa_location": {"pdf_url": "https://example.test/html-first.pdf"},
            "locations": [
                {"pdf_url": "https://example.test/oversize.pdf"},
                {"pdf_url": "https://example.test/first.pdf"},
            ],
        },
        {
            "id": "https://openalex.org/W2",
            "doi": "https://doi.org/10.1/two",
            "display_name": "Second",
            "is_retracted": False,
            "open_access": {"is_oa": True},
            "best_oa_location": {"pdf_url": "https://example.test/second.pdf"},
            "locations": [],
        },
    ]
    calls = []

    def fake_request(url, **kwargs):
        calls.append((url, kwargs))
        if "api.openalex.org" in url:
            return {
                "status": 200,
                "headers": {"x-ratelimit-limit": "1000", "x-ratelimit-remaining": "34"},
                "json": {"meta": {"count": 2}, "results": candidates},
            }
        if url.endswith("html-first.pdf"):
            return {"status": 200, "headers": {"Content-Type": "text/html"}, "body": b"<html>no pdf</html>"}
        if url.endswith("oversize.pdf"):
            return {"status": 200, "headers": {}, "body": b"", "oversize": True}
        if url.endswith("first.pdf"):
            return {"status": 200, "headers": {}, "body": pdfs[0]}
        return {"status": 200, "headers": {}, "body": pdfs[1]}

    monkeypatch.setattr(retrieval_module, "_paper_search_http_request", fake_request)
    first_paths = agent.search_papers(
        ["open query"],
        str(paper_dir),
        target_papers=1,
        provider_strategy="openalex",
        keyword_delay_seconds=0,
        download_delay_seconds=0,
    )
    second_paths = agent.search_papers(
        ["open query"],
        str(paper_dir),
        target_papers=2,
        provider_strategy="openalex",
        keyword_delay_seconds=0,
        download_delay_seconds=0,
    )
    assert len(first_paths) == 1
    assert len(second_paths) == 2
    assert not stale.exists()
    assert len(list(paper_dir.glob("*.pdf"))) == 2
    assert any(
        item.get("failure") == "invalid_pdf"
        for item in agent._paper_search_audit["download_attempts"]
    )
    assert any(
        item.get("failure") == "oversize"
        for item in agent._paper_search_audit["download_attempts"]
    )
    oa_call = next(item for item in calls if "api.openalex.org" in item[0])
    assert oa_call[1]["params"]["per_page"] == 100
    assert "open_access.is_oa:true" in oa_call[1]["params"]["filter"]
    assert "has_pdf_url:true" in oa_call[1]["params"]["filter"]


def test_openalex_key_is_header_and_anonymous_429_is_not_retried(tmp_path, monkeypatch):
    agent = make_test_retrieval_agent(deepseek_api_key="")
    observed = []
    monkeypatch.setenv("OPENALEX_API_KEY", "secret-test-key")

    def fake_key_request(url, **kwargs):
        observed.append((url, kwargs))
        return {"status": 429, "headers": {"Retry-After": "4"}, "body": b"limited"}

    monkeypatch.setattr(retrieval_module, "_paper_search_http_request", fake_key_request)
    agent.search_papers(
        ["key query"],
        str(tmp_path / "papers"),
        target_papers=1,
        provider_strategy="openalex",
        keyword_delay_seconds=0,
        download_delay_seconds=0,
    )
    assert len(observed) == 3  # API key allows bounded 429 retries.
    assert all("api_key" not in item[1]["params"] for item in observed)
    assert all(item[1]["headers"]["Authorization"] == "Bearer secret-test-key" for item in observed)
    assert "secret-test-key" not in json.dumps(agent._paper_search_audit)

    monkeypatch.delenv("OPENALEX_API_KEY")
    observed.clear()
    agent.search_papers(
        ["anonymous query"],
        str(tmp_path / "anonymous"),
        target_papers=1,
        provider_strategy="openalex",
        keyword_delay_seconds=0,
        download_delay_seconds=0,
    )
    assert len(observed) == 1
    assert agent._paper_search_audit["provider_attempts"][0]["shared_pool_limited"] is True


def test_openalex_key_retries_5xx_with_bounded_backoff(tmp_path, monkeypatch):
    agent = make_test_retrieval_agent(deepseek_api_key="")
    monkeypatch.setenv("OPENALEX_API_KEY", "secret-test-key")
    responses = [
        {"status": 503, "headers": {"Retry-After": "2"}, "body": b"busy"},
        {"status": 503, "headers": {}, "body": b"busy"},
        {"status": 200, "headers": {}, "json": {"results": []}},
    ]
    calls = []

    def fake_request(url, **kwargs):
        calls.append((url, kwargs))
        return responses.pop(0)

    monkeypatch.setattr(retrieval_module, "_paper_search_http_request", fake_request)
    sleeps = []
    monkeypatch.setattr(agent, "_sleep_paper_search_with_backoff", sleeps.append)
    agent.search_papers(
        ["5xx query"],
        str(tmp_path / "papers"),
        target_papers=1,
        provider_strategy="openalex",
        keyword_delay_seconds=0,
        download_delay_seconds=0,
    )
    assert len(calls) == 3
    assert sleeps[:2] == [2, 4]
    assert agent._paper_search_audit["provider_attempts"][0]["retry_count"] == 2


def test_openalex_budget_shortage_stops_before_next_request(tmp_path, monkeypatch):
    agent = make_test_retrieval_agent(deepseek_api_key="")
    calls = []

    def fake_request(url, **kwargs):
        calls.append(url)
        return {
            "status": 200,
            "headers": {"x-ratelimit-remaining": "5"},
            "json": {"results": []},
        }

    monkeypatch.setattr(retrieval_module, "_paper_search_http_request", fake_request)
    agent.search_papers(
        ["one", "two"],
        str(tmp_path / "papers"),
        target_papers=2,
        provider_strategy="openalex",
        keyword_delay_seconds=0,
        download_delay_seconds=0,
    )
    assert len(calls) == 1
    assert agent._paper_search_audit["provider_attempts"][1]["skipped_reason"] == "provider_budget_insufficient"


def test_provider_keyword_timeout_continues_to_next_keyword_without_downstream(
    tmp_path, monkeypatch
):
    """首关键词下载超时只标记该关键词，下一关键词仍可成功返回。"""

    agent = make_test_retrieval_agent(deepseek_api_key="")
    keyword_calls = []
    download_calls = []
    statuses = []

    def fake_candidates(keyword, **kwargs):
        keyword_calls.append(keyword)
        slug = keyword.replace(" ", "-")
        return [{"id": f"W-{slug}", "display_name": keyword, "best_oa_location": {"pdf_url": f"https://example.test/{slug}.pdf"}}]

    def fake_rank(candidates, question, **kwargs):
        return candidates

    def fake_request(url, **kwargs):
        download_calls.append(url)
        if url.endswith("first-timeout.pdf"):
            raise PaperSearchTimeoutError("下载阶段达到关键词墙钟上限")
        return {"status": 200, "headers": {}, "body": _make_pdf_bytes("second success")}

    monkeypatch.setattr(agent, "_request_openalex_candidates", fake_candidates)
    monkeypatch.setattr(agent, "_rank_openalex_candidates_locally", fake_rank)
    monkeypatch.setattr(retrieval_module, "_paper_search_http_request", fake_request)

    paths = agent.search_papers(
        ["first timeout", "second success"],
        str(tmp_path / "papers"),
        target_papers=1,
        timeout_seconds=10,
        provider_strategy="openalex",
        keyword_delay_seconds=0,
        download_delay_seconds=0,
        status_sink=statuses.append,
    )
    assert keyword_calls == ["first timeout", "second success"]
    assert [url.rsplit("/", 1)[-1] for url in download_calls] == [
        "first-timeout.pdf",
        "second-success.pdf",
    ]
    assert [status["status"] for status in statuses] == ["timeout", "success"]
    assert statuses[0]["timeout_seconds"] == 10
    assert "下载阶段达到关键词墙钟上限" in statuses[0]["error"]
    assert len(paths) == 1


def test_anonymous_semantic_scholar_success_continues_until_first_429(tmp_path, monkeypatch):
    agent = make_test_retrieval_agent(deepseek_api_key="")
    pdf_bytes = _make_pdf_bytes("anonymous s2")
    responses = [
        {
            "status": 200,
            "headers": {},
            "json": {
                "data": [
                    {
                        "paperId": "anonymous-1",
                        "title": "One",
                        "openAccessPdf": {"url": "https://example.test/one.pdf"},
                    }
                ]
            },
        },
        {"status": 429, "headers": {"Retry-After": "120"}, "body": b"limited"},
    ]
    calls = []

    def fake_request(url, **kwargs):
        calls.append(url)
        if url.endswith("one.pdf"):
            return {"status": 200, "headers": {}, "body": pdf_bytes}
        return responses.pop(0)

    monkeypatch.setattr(retrieval_module, "_paper_search_http_request", fake_request)
    sleeps = []
    monkeypatch.setattr(agent, "_sleep_paper_search_with_backoff", sleeps.append)
    paths = agent.search_papers(
        ["first", "second", "third"],
        str(tmp_path / "papers"),
        target_papers=2,
        provider_strategy="semantic_scholar",
        keyword_delay_seconds=0,
        download_delay_seconds=0,
    )
    assert len(paths) == 1
    s2_calls = [url for url in calls if "semanticscholar" in url]
    assert len(s2_calls) == 2
    assert any(delay > 0 for delay in sleeps)
    assert agent._paper_search_audit["provider_attempts"][2]["skipped_reason"] == "anonymous_shared_pool_limited"


def test_semantic_scholar_key_enforces_one_request_per_second_across_keywords(
    tmp_path, monkeypatch
):
    agent = make_test_retrieval_agent(deepseek_api_key="")
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "secret-test-key")
    pdfs = [_make_pdf_bytes("one"), _make_pdf_bytes("two")]
    calls = []

    def fake_request(url, **kwargs):
        calls.append(url)
        if url.endswith("one.pdf"):
            return {"status": 200, "headers": {}, "body": pdfs[0]}
        if url.endswith("two.pdf"):
            return {"status": 200, "headers": {}, "body": pdfs[1]}
        index = len([item for item in calls if "semanticscholar" in item])
        return {
            "status": 200,
            "headers": {},
            "json": {
                "data": [
                    {
                        "paperId": f"key-{index}",
                        "title": f"Key {index}",
                        "openAccessPdf": {"url": f"https://example.test/{['one', 'two'][index - 1]}.pdf"},
                    }
                ]
            },
        }

    monkeypatch.setattr(retrieval_module, "_paper_search_http_request", fake_request)
    sleeps = []
    monkeypatch.setattr(agent, "_sleep_paper_search_with_backoff", sleeps.append)
    paths = agent.search_papers(
        ["one", "two"],
        str(tmp_path / "papers"),
        target_papers=2,
        provider_strategy="semantic_scholar",
        keyword_delay_seconds=0,
        download_delay_seconds=0,
    )
    assert len(paths) == 2
    assert any(delay > 0 for delay in sleeps)
