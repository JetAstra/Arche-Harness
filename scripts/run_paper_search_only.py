#!/usr/bin/env python3
"""只执行 Retrieval 的关键词抽取与 PaperSearch 下载。

脚本把论文、检索 manifest、恢复状态和响应缓存写入调用方指定的目录；不会构建
向量索引，也不会启动 Hypothesis、Investigation 或 MiniChem。
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Mapping, Optional, Sequence

from chemistry_multiagent.agents.retrieval_agent import RetrievalAgent


def build_paper_search_only_parser() -> argparse.ArgumentParser:
    """声明 PaperSearch-only 入口参数，不暴露后续索引和工作流阶段。"""

    parser = argparse.ArgumentParser(
        description="抽取关键词并下载论文；执行到 PaperSearch 后立即停止",
    )
    parser.add_argument("--question", required=True, help="原始科学问题，两段文本应原样传入")
    parser.add_argument("--model-path", help="本地 BGE SentenceTransformer 模型目录")
    parser.add_argument(
        "--pdf-dir",
        default="local_data/papers/case1-paper-search/papers",
        help="下载 PDF 的本地目录",
    )
    parser.add_argument(
        "--manifest",
        default="local_data/papers/case1-paper-search/manifest.json",
        help="脱敏检索记录路径",
    )
    parser.add_argument(
        "--candidate-page-size",
        type=int,
        default=100,
        help="兼容 Semantic Scholar 每页候选数量；Scholar provider 不使用此参数",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=180.0,
        help="单个关键词 PaperSearch 墙钟上限，默认 180 秒",
    )
    parser.add_argument(
        "--target-papers",
        type=int,
        default=50,
        help="全局去重 PDF 目标；这是本仓库运行策略，默认50",
    )
    parser.add_argument(
        "--keyword-delay-seconds",
        type=float,
        default=45.0,
        help="关键词间退避秒数，默认45秒",
    )
    parser.add_argument(
        "--download-delay-seconds",
        type=float,
        default=1.0,
        help="开放获取 PDF 下载之间的最小间隔，默认1秒",
    )
    parser.add_argument(
        "--openalex-per-page",
        type=int,
        default=100,
        help="OpenAlex 单次元数据候选数（1–100），默认100",
    )
    parser.add_argument(
        "--scholar-max-pages",
        type=int,
        default=3,
        help="Google Scholar 每关键词最多读取页数（1–20），默认3",
    )
    parser.add_argument(
        "--scholar-results-per-page",
        type=int,
        default=10,
        help="Google Scholar 每页结果数（1–20），默认10",
    )
    parser.add_argument(
        "--scholar-crossref-max-candidates",
        type=int,
        default=3,
        help="每个 Scholar 页最多用 Crossref 补充 DOI 的候选数（0–20），默认3",
    )
    parser.add_argument(
        "--scholar-request-delay-seconds",
        type=float,
        default=None,
        help="Scholar 页面请求之间的最小等待秒数，默认15秒",
    )
    parser.add_argument(
        "--scholar-request-jitter-seconds",
        type=float,
        default=None,
        help="Scholar 页面请求前的随机抖动上限，默认15秒",
    )
    parser.add_argument(
        "--scholar-initial-delay-seconds",
        type=float,
        default=None,
        help="首次 Scholar 请求前的预等待秒数，默认10秒",
    )
    parser.add_argument(
        "--scholar-initial-jitter-seconds",
        type=float,
        default=None,
        help="首次 Scholar 请求前的随机抖动上限，默认20",
    )
    parser.add_argument(
        "--scholar-backoff-base-seconds",
        type=float,
        default=None,
        help="Scholar 403/429 指数退避基础秒数，默认900秒",
    )
    parser.add_argument(
        "--scholar-max-backoff-seconds",
        type=float,
        default=None,
        help="Scholar 单次限流冷却上限，默认3600秒",
    )
    parser.add_argument(
        "--scholar-max-consecutive-rate-limits",
        type=int,
        default=None,
        help="Scholar 连续403/429 后停机次数，默认1",
    )
    parser.add_argument(
        "--keyword-jitter-seconds",
        type=float,
        default=None,
        help="关键词间随机抖动上限，默认45秒",
    )
    parser.add_argument(
        "--resume-state",
        default=None,
        help="关键词、页码和已接受文献恢复文件；默认位于 PDF 目录",
    )
    parser.add_argument(
        "--response-cache-dir",
        default=None,
        help="Scholar 合法 HTML 响应缓存目录；默认位于 PDF 目录",
    )
    parser.add_argument(
        "--provider-strategy",
        choices=(
            "pypaperbot_scholar",
            "paper_scraper",
            "semantic_scholar_then_openalex",
            "semantic_scholar",
            "openalex",
        ),
        default="pypaperbot_scholar",
        help="PaperSearch provider；默认受控 Google Scholar，其他值仅兼容旧路径",
    )
    parser.add_argument(
        "--keywords-json",
        help="可选的已冻结关键词 JSON 数组；提供后不再次调用关键词模型",
    )
    return parser


def run_paper_search_only_from_args(
    argv: Optional[Sequence[str]] = None,
) -> Mapping[str, Any]:
    """解析参数，运行一次 PaperSearch-only，并返回 manifest 内容。"""

    parser = build_paper_search_only_parser()
    args = parser.parse_args(argv)
    if args.candidate_page_size < 1:
        parser.error("--candidate-page-size 必须为正整数")
    if args.timeout_seconds is not None and args.timeout_seconds <= 0:
        parser.error("--timeout-seconds 必须为正数")
    if args.target_papers <= 0:
        parser.error("--target-papers 必须为正整数")
    if args.keyword_delay_seconds < 0:
        parser.error("--keyword-delay-seconds 不能为负数")
    if args.download_delay_seconds < 0:
        parser.error("--download-delay-seconds 不能为负数")
    if not 1 <= args.openalex_per_page <= 100:
        parser.error("--openalex-per-page 必须在1到100之间")
    if not 1 <= args.scholar_max_pages <= 20:
        parser.error("--scholar-max-pages 必须在1到20之间")
    if not 1 <= args.scholar_results_per_page <= 20:
        parser.error("--scholar-results-per-page 必须在1到20之间")
    if not 0 <= args.scholar_crossref_max_candidates <= 20:
        parser.error("--scholar-crossref-max-candidates 必须在0到20之间")
    for option_name in (
        "scholar_request_delay_seconds",
        "scholar_request_jitter_seconds",
        "scholar_initial_delay_seconds",
        "scholar_initial_jitter_seconds",
        "scholar_backoff_base_seconds",
        "scholar_max_backoff_seconds",
        "keyword_jitter_seconds",
    ):
        option_value = getattr(args, option_name)
        if option_value is not None and option_value < 0:
            parser.error(f"--{option_name.replace('_', '-')} 不能为负数")
    if (
        args.scholar_max_consecutive_rate_limits is not None
        and args.scholar_max_consecutive_rate_limits < 1
    ):
        parser.error("--scholar-max-consecutive-rate-limits 必须为正整数")
    keywords_override = None
    if args.keywords_json is not None:
        try:
            keywords_override = json.loads(args.keywords_json)
        except json.JSONDecodeError as exc:
            parser.error(f"--keywords-json 不是严格 JSON：{exc.msg}")
        if (
            not isinstance(keywords_override, list)
            or not keywords_override
            or any(not isinstance(item, str) or not item.strip() for item in keywords_override)
        ):
            parser.error("--keywords-json 必须是非空字符串数组")
    agent = RetrievalAgent(model_path=args.model_path)
    return agent.run_paper_search_only(
        question=args.question,
        pdf_dir=args.pdf_dir,
        manifest_path=args.manifest,
        candidate_page_size=args.candidate_page_size,
        timeout_seconds=args.timeout_seconds,
        target_papers=args.target_papers,
        keyword_delay_seconds=args.keyword_delay_seconds,
        keywords_override=keywords_override,
        provider_strategy=args.provider_strategy,
        openalex_per_page=args.openalex_per_page,
        download_delay_seconds=args.download_delay_seconds,
        scholar_max_pages=args.scholar_max_pages,
        scholar_results_per_page=args.scholar_results_per_page,
        scholar_crossref_max_candidates=args.scholar_crossref_max_candidates,
        scholar_request_delay_seconds=args.scholar_request_delay_seconds,
        scholar_request_jitter_seconds=args.scholar_request_jitter_seconds,
        scholar_initial_delay_seconds=args.scholar_initial_delay_seconds,
        scholar_initial_jitter_seconds=args.scholar_initial_jitter_seconds,
        scholar_backoff_base_seconds=args.scholar_backoff_base_seconds,
        scholar_max_backoff_seconds=args.scholar_max_backoff_seconds,
        scholar_max_consecutive_rate_limits=args.scholar_max_consecutive_rate_limits,
        keyword_jitter_seconds=args.keyword_jitter_seconds,
        resume_state_path=args.resume_state,
        response_cache_dir=args.response_cache_dir,
    )


def main() -> int:
    """运行入口并打印脱敏 manifest，不打印环境变量或模型服务密钥。"""

    manifest = run_paper_search_only_from_args()
    print(json.dumps(dict(manifest), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
