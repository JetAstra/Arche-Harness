#!/usr/bin/env python3
"""
Retrieval Agent - 检索智能体

功能：
1. 解析用户科学问题，提取关键词
2. 检索相关学术论文
3. 构建论文索引
4. 提供背景信息和回答用户问题
5. 为后续假设生成Agent提供文献综述

整合自：
- extract_keywords_for_search_paper.py
- search.py
"""

import os
import json
import re
import logging
import time
import base64
import random
import sys
import signal
import threading
import asyncio
import hashlib
import tempfile
import uuid
import ipaddress
import socket
from pathlib import PurePosixPath
from html.parser import HTMLParser
from urllib.parse import urlsplit
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from difflib import SequenceMatcher
from importlib import metadata as importlib_metadata
from contextlib import contextmanager
import numpy as np
from copy import deepcopy
from typing import List, Dict, Tuple, Optional, Any, Callable, Mapping, Sequence

# SentenceTransformer 必须先于 PyMuPDF/FAISS 导入，避免这些 C++ 扩展先加载系统
# libstdc++，导致 SciPy/sklearn 的语义嵌入依赖无法解析 ABI。
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    SENTENCE_TRANSFORMERS_IMPORT_ERROR = None
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SENTENCE_TRANSFORMERS_IMPORT_ERROR = "sentence-transformers 未安装"

# PDF处理
import fitz  # PyMuPDF

try:
    import requests
except ImportError:  # pragma: no cover - 标准环境提供 requests，保留明确失败信息
    requests = None

# 论文爬取
try:
    import paperscraper
    PAPERSCRAPER_AVAILABLE = True
except ImportError:
    paperscraper = None
    PAPERSCRAPER_AVAILABLE = False
    print("警告: paperscraper 不可用，将跳过论文下载功能")

# 向量检索。必须在 SentenceTransformer 导入后加载，避免底层 C++ 运行库先被
# FAISS 锁定为系统版本而导致 SciPy/sklearn 无法加载。
import faiss

# Deepseek API
import openai
from jsonschema import Draft202012Validator

from chemistry_multiagent.utils.model_operation import (
    FAILURE_KINDS,
    ModelStage,
    PROTOCOL_RESPONSE_LIMIT,
    extract_gaussian_expert_text,
    normalize_provider_envelope,
    validate_model_envelope,
)
from chemistry_multiagent.utils.llm_api import call_compiled_model_request
from chemistry_multiagent.utils.api_retry import (
    MODEL_CALL_RETRY_COUNT,
    MODEL_CALL_RETRY_DELAY_SECONDS,
    is_retryable_http_status,
    response_http_status,
    retry_api_call,
    suppress_nested_api_retries,
)
from chemistry_multiagent.utils.model_catalog import (
    compile_model_request,
)
from chemistry_multiagent.utils.model_defaults import resolve_general_model
from chemistry_multiagent.utils.model_prompts import build_model_messages
from chemistry_multiagent.utils.scientific_input import (
    merge_supplementary_chemistry_context,
    normalize_supplementary_scientific_input,
    project_supplementary_scientific_input_from_context,
)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@contextmanager
def _enable_paperscraper_proxy_from_environment(*, enable_proxy: bool = True):
    """让 paperscraper 的 aiohttp 会话读取标准代理环境变量。

    paperscraper v1.8.1 创建 ``aiohttp.ClientSession`` 时默认关闭 ``trust_env``，
    因而仅设置 HTTP(S)_PROXY 也不会经过本机代理。此上下文只在代理变量存在时
    临时替换会话类，退出后恢复第三方模块原对象，不修改 vendored 或安装包文件。
    """

    if not enable_proxy:
        yield
        return
    proxy_configured = any(
        os.environ.get(name)
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
    )
    paperscraper_lib = getattr(paperscraper, "lib", None)
    if paperscraper_lib is None:
        yield
        return
    session_class = getattr(paperscraper_lib, "ThrottledClientSession", None)

    original_session_classes = []
    original_async_search = getattr(paperscraper_lib, "a_search_papers", None)

    proxy_session_class = None
    if session_class is not None:
        class ProxyAwareThrottledClientSession(session_class):
            """临时开启 aiohttp 对 HTTP(S)_PROXY 的读取。"""

            def __init__(self, *args, **kwargs):
                kwargs.setdefault("trust_env", True)
                super().__init__(*args, **kwargs)

        proxy_session_class = ProxyAwareThrottledClientSession

    modules_to_patch = [paperscraper_lib]
    paperscraper_scraper = sys.modules.get("paperscraper.scraper")
    if proxy_configured and paperscraper_scraper is not None:
        modules_to_patch.append(paperscraper_scraper)
    if proxy_configured and session_class is not None:
        for module in modules_to_patch:
            module_session_class = getattr(module, "ThrottledClientSession", None)
            if module_session_class is not None:
                original_session_classes.append((module, module_session_class))
                module.ThrottledClientSession = proxy_session_class

    if original_async_search is not None:
        async def close_failed_search_sessions(*args, **kwargs):
            """确保 paperscraper 搜索异常时关闭其临时 aiohttp 会话。"""

            scraper = kwargs.get("scraper")
            owns_scraper = scraper is None
            if owns_scraper:
                scraper = paperscraper_lib.default_scraper()
                kwargs["scraper"] = scraper
            succeeded = False
            try:
                result = await original_async_search(*args, **kwargs)
                succeeded = True
                return result
            finally:
                if owns_scraper and not succeeded:
                    try:
                        await scraper.close()
                    except (Exception, asyncio.CancelledError):
                        logger.debug("关闭失败的 paperscraper 会话时忽略清理异常")

        paperscraper_lib.a_search_papers = close_failed_search_sessions
    try:
        yield
    finally:
        if original_async_search is not None:
            paperscraper_lib.a_search_papers = original_async_search
        for module, module_session_class in original_session_classes:
            module.ThrottledClientSession = module_session_class


_ELEMENT_SYMBOLS = frozenset(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe "
    "Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg "
    "Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg "
    "Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og".split()
)


# ==================== 检索协议与错误 ====================

def _retrieval_protocol_error(history: List[Mapping[str, str]]) -> Dict[str, str]:
    """把本阶段已经出现的不同错误一并交给下一次完整重答。"""

    current = history[-1]
    current_pair = (current["failure_kind"], current["summary"])
    earlier = []
    seen = set()
    for item in history[:-1]:
        pair = (item["failure_kind"], item["summary"])
        if pair == current_pair or pair in seen:
            continue
        seen.add(pair)
        earlier.append(f"{pair[0]}: {pair[1]}")
    summary = current_pair[1]
    if earlier:
        summary += "; earlier stage failures must not recur: " + " | ".join(
            earlier[:3]
        )
    return {"failure_kind": current_pair[0], "summary": summary}


class RetrievalRootError(RuntimeError):
    """检索流程无法形成可信科学输入时，明确终止本次根操作。

    上层收到该错误后应暂停整个假设生成阶段。空关键词、空回答或截断综述都不会
    被问题文字或默认模板替代，否则后续闭环看似运行，实际却没有可靠证据起点。
    """

    def __init__(self, stage: str, failure_kind: str):
        self.stage = stage
        self.failure_kind = failure_kind
        super().__init__(f"{stage} stage 失败：{failure_kind}")


class RetrievalEmbeddingError(RuntimeError):
    """语义嵌入不可用时终止检索，避免悄悄改用词频匹配。"""


class PaperSearchTimeoutError(RuntimeError):
    """单关键词超过墙钟上限时中断当前阶段，避免被 HTTP 库当作套接字超时吞掉。"""


SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_SEARCH_URL = "https://api.openalex.org/works"
GOOGLE_SCHOLAR_SEARCH_URL = "https://scholar.google.com/scholar"
CROSSREF_WORKS_URL = "https://api.crossref.org/works"

# Scholar 结果中的 PDF 链接只允许公开页面明确提供的 http(s) 地址。下面的域名
# 片段用于阻断影子文献库；它们不参与搜索，也不会被写入 manifest。
_BLOCKED_PAPER_HOST_FRAGMENTS = (
    "sci-hub",
    "scihub",
    "annas-archive",
    "annasarchive",
    "sci_db",
    "scidb",
)

# HTML 规范中的 void 元素没有结束标签。Google Scholar 的实际响应通常把
# ``meta``、``input`` 等元素写成未闭合的开始标签；解析器不能把它们计入结果
# 容器深度，否则外层 ``div.gs_r`` 的结束标签永远无法与开始深度匹配。
_HTML_VOID_TAGS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)


def _paper_search_http_request(
    url: str,
    *,
    params: Optional[Mapping[str, Any]] = None,
    headers: Optional[Mapping[str, str]] = None,
    timeout_seconds: float = 30.0,
    max_bytes: int = 8 * 1024 * 1024,
) -> Dict[str, Any]:
    """发送一次论文服务 HTTP 请求并返回脱敏响应。

    输入：服务 URL、查询参数、请求头和墙钟超时。动作：通过 ``requests`` 读取当前
    进程代理设置，返回状态、响应头、正文和可选 JSON。副作用：产生一次外部网络请求。
    返回：供 provider 层消费的普通字典；失败：依赖缺失或传输异常原样抛出。请求头
    和响应正文不会写入 manifest，调用方不得把其中的密钥复制到日志。
    """

    if requests is None:
        raise RuntimeError("PaperSearch 需要 requests 依赖")
    response = requests.get(
        url,
        params=dict(params or {}),
        headers=dict(headers or {}),
        timeout=timeout_seconds,
        stream=True,
        allow_redirects=False,
    )
    response_headers = {str(k): str(v) for k, v in response.headers.items()}
    lowered_response_headers = {
        key.lower(): value for key, value in response_headers.items()
    }
    try:
        declared_length = int(lowered_response_headers.get("content-length", "0"))
    except ValueError:
        declared_length = 0
    if declared_length > max_bytes:
        response.close()
        return {
            "status": int(response.status_code),
            "headers": response_headers,
            "body": b"",
            "oversize": True,
        }
    chunks: List[bytes] = []
    total_bytes = 0
    oversize = False
    try:
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                oversize = True
                break
            chunks.append(bytes(chunk))
    finally:
        response.close()
    body = b"".join(chunks)
    status_code = int(response.status_code)
    result: Dict[str, Any] = {
        "status": status_code,
        "headers": response_headers,
        "body": body,
    }
    if oversize:
        result["oversize"] = True
    content_type = lowered_response_headers.get("content-type", "").lower()
    if not oversize and "json" in content_type:
        try:
            result["json"] = json.loads(body.decode("utf-8"))
        except (TypeError, ValueError, UnicodeDecodeError):
            pass
    return result


def _normalize_paper_search_http_response(response: Any) -> Dict[str, Any]:
    """把真实响应或测试替身归一化为状态、头、正文和 JSON 字段。"""

    if isinstance(response, Mapping):
        normalized = dict(response)
        normalized["status"] = int(normalized.get("status", normalized.get("status_code", 0)))
        normalized["headers"] = {
            str(k): str(v) for k, v in dict(normalized.get("headers") or {}).items()
        }
        return normalized
    status = int(getattr(response, "status_code", getattr(response, "status", 0)))
    headers = {
        str(k): str(v)
        for k, v in dict(getattr(response, "headers", {}) or {}).items()
    }
    body = getattr(response, "content", b"") or b""
    normalized = {"status": status, "headers": headers, "body": body}
    json_method = getattr(response, "json", None)
    if callable(json_method):
        try:
            normalized["json"] = json_method()
        except (TypeError, ValueError):
            pass
    return normalized


def _paper_search_response_cache_key(
    url: str,
    params: Optional[Mapping[str, Any]] = None,
) -> str:
    """为合法的论文元数据响应生成稳定缓存键。

    输入：请求 URL 和查询参数。动作：按排序后的 JSON 表示计算 SHA-256；不发起
    网络请求，也不把请求头或密钥写入缓存。返回：可作为缓存文件名的十六进制键。
    """

    payload = json.dumps(
        {"url": str(url), "params": dict(params or {})},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_paper_search_response_cache(
    cache_dir: Optional[str],
    cache_key: str,
) -> Optional[Dict[str, Any]]:
    """读取脱敏的论文响应缓存，损坏或缺失时返回 ``None``。

    缓存只保存状态码和正文，不保存请求头、Cookie 或 API 密钥；这样进程重启后
    可以重用已经取得的 Scholar HTML，又不会把认证信息带入 manifest。
    """

    if not cache_dir:
        return None
    cache_path = os.path.join(os.path.abspath(cache_dir), f"{cache_key}.json")
    try:
        with open(cache_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            return None
        body = base64.b64decode(str(payload.get("body_b64", "")), validate=True)
        return {
            "status": int(payload.get("status", 0)),
            "headers": {},
            "body": body,
        }
    except (OSError, TypeError, ValueError, base64.binascii.Error):
        return None


def _write_paper_search_response_cache(
    cache_dir: Optional[str],
    cache_key: str,
    response: Mapping[str, Any],
) -> None:
    """原子写入一个合法的论文响应缓存。

    输入：缓存目录、请求键和已归一化响应。动作：只缓存 200 响应的正文，并通过
    临时文件加 ``os.replace`` 防止进程中断留下半份缓存。失败：缓存写入失败仅记录
    调试日志，不阻断当前 Scholar 请求。
    """

    if not cache_dir or int(response.get("status", 0)) != 200:
        return
    body = response.get("body", b"")
    if not isinstance(body, (bytes, bytearray)):
        body = str(body).encode("utf-8", errors="ignore")
    cache_root = os.path.abspath(cache_dir)
    temporary_path = ""
    try:
        os.makedirs(cache_root, exist_ok=True)
        cache_path = os.path.join(cache_root, f"{cache_key}.json")
        fd, temporary_path = tempfile.mkstemp(
            prefix=f".{cache_key}.", suffix=".part", dir=cache_root
        )
        payload = {
            "schema_version": 1,
            "status": 200,
            "body_b64": base64.b64encode(bytes(body)).decode("ascii"),
        }
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, cache_path)
        temporary_path = ""
    except OSError:
        logger.debug("写入 PaperSearch 响应缓存失败", exc_info=True)
    finally:
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass


def _is_allowed_public_pdf_url(url: Any) -> bool:
    """判断 PDF 地址是否是可访问的公开 HTTP(S) 地址。

    输入：候选记录中的 URL。动作：解析传输协议和主机名，并拒绝影子文献库域名。
    返回：仅当地址可作为公开 PDF 下载候选时返回 ``True``。失败：无法解析、协议不
    是 HTTP(S) 或命中阻断域名时返回 ``False``；本函数不会发起网络请求。
    """

    if not isinstance(url, str) or not url.strip():
        return False
    try:
        parsed = urlsplit(url.strip())
    except ValueError:
        return False
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return False
    host = parsed.hostname.lower().rstrip(".")
    if host in {"localhost", "localhost.localdomain", "ip6-localhost"}:
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is None:
        try:
            resolved = {
                ipaddress.ip_address(item[4][0])
                for item in socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)
                if item[4] and item[4][0]
            }
        except (OSError, ValueError):
            resolved = set()
        if any(
            item.is_loopback
            or item.is_private
            or item.is_link_local
            or item.is_reserved
            or item.is_multicast
            or item.is_unspecified
            for item in resolved
        ):
            return False
    if address is not None and (
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    ):
        return False
    lowered_url = url.strip().lower()
    return not any(
        fragment in host or fragment in lowered_url
        for fragment in _BLOCKED_PAPER_HOST_FRAGMENTS
    )


class _GoogleScholarResultParser(HTMLParser):
    """解析 Google Scholar 结果卡片中可见的元数据和公开 PDF 链接。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._depth = 0
        self._tag_stack: List[Tuple[str, bool]] = []
        self._result_root: Optional[Tuple[str, int]] = None
        self._current: Optional[Dict[str, str]] = None
        self.results: List[Dict[str, str]] = []
        self._capture_field: Optional[str] = None
        self._capture_tag: Optional[str] = None
        self._capture_depth: Optional[int] = None
        self._capture_chunks: List[str] = []
        self._pdf_block_depth = 0

    @staticmethod
    def _classes(attrs: Mapping[str, Optional[str]]) -> set[str]:
        return set(str(attrs.get("class") or "").split())

    @staticmethod
    def _is_result_container(tag: str, classes: set[str]) -> bool:
        """判断元素是否可能是包住一条 Scholar 结果的外层容器。"""

        if tag not in {"div", "td", "li"}:
            return False
        return bool(classes & {"gs_r", "gs_or", "gs_scl", "gs_ri"})

    def _begin_capture(self, field: str, tag: str) -> None:
        if self._current is None or self._capture_field is not None:
            return
        self._capture_field = field
        self._capture_tag = tag
        self._capture_depth = self._depth
        self._capture_chunks = []

    def _finish_capture(self) -> None:
        if self._current is None or self._capture_field is None:
            return
        value = " ".join("".join(self._capture_chunks).split())
        if value:
            self._current[self._capture_field] = value
        self._capture_field = None
        self._capture_tag = None
        self._capture_depth = None
        self._capture_chunks = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        normalized_tag = tag.lower()
        is_void_tag = normalized_tag in _HTML_VOID_TAGS
        if not is_void_tag:
            self._depth += 1
        attributes = {key: value for key, value in attrs}
        classes = self._classes(attributes)
        is_pdf_block = bool(classes & {"gs_ggs", "gs_ggsd", "gs_or_ggsm"})
        if not is_void_tag:
            self._tag_stack.append((tag, is_pdf_block))
        if self._current is None and self._is_result_container(tag, classes):
            self._result_root = (tag, self._depth)
            self._current = {}
            scholar_id = attributes.get("data-cid")
            if scholar_id:
                self._current["scholar_id"] = scholar_id.strip()
        if self._current is None:
            return
        if is_pdf_block:
            self._pdf_block_depth += 1
        if tag == "h3" and "gs_rt" in classes:
            self._begin_capture("title", tag)
            return
        if tag == "div" and "gs_a" in classes:
            self._begin_capture("authors", tag)
            return
        if tag == "div" and "gs_rs" in classes:
            self._begin_capture("snippet", tag)
            return
        if tag == "a":
            href = attributes.get("href")
            if self._capture_field == "title" and href:
                self._current["result_url"] = href.strip()
            if self._pdf_block_depth > 0 and href and _is_allowed_public_pdf_url(href):
                self._current.setdefault("scholar_pdf_url", href.strip())

    def handle_startendtag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        if self._capture_field is not None:
            self._capture_chunks.append(data)

    def handle_endtag(self, tag: str) -> None:
        # HTML void 元素允许省略结束标签；如果响应恰好带有 ``</input>`` 等
        # 非标准结束标签，也不能让它弹出前面的结果容器栈。
        if tag.lower() in _HTML_VOID_TAGS:
            return
        if self._capture_tag == tag and self._capture_depth == self._depth:
            self._finish_capture()
        result_root = self._result_root
        if (
            result_root is not None
            and self._current is not None
            and result_root == (tag, self._depth)
        ):
            self._finish_capture()
            if self._current.get("title"):
                self.results.append(dict(self._current))
            self._current = None
            self._result_root = None
            self._pdf_block_depth = 0
        if self._tag_stack:
            _, is_pdf_block = self._tag_stack.pop()
            if is_pdf_block and self._pdf_block_depth > 0:
                self._pdf_block_depth -= 1
        self._depth = max(0, self._depth - 1)


def _parse_google_scholar_html(html: Any) -> List[Dict[str, str]]:
    """从 Scholar HTML 中提取结果卡片，不解析或构造付费下载链接。

    输入：Google Scholar 返回的 UTF-8 HTML 字节或字符串。动作：以 ``gs_r``、``gs_or``
    或 ``gs_scl`` 等外层结果容器关联 ``gs_ri`` 元数据和同级 ``gs_ggs`` 公开 PDF 链接，
    也兼容 PDF 链接嵌在 ``gs_ri`` 内的页面。返回：候选字典列表；
    解析失败时返回空列表，不会抛出交互式验证码等待。失败：非文本正文按空结果处理。
    """

    if isinstance(html, bytes):
        text = html.decode("utf-8", errors="replace")
    elif isinstance(html, str):
        text = html
    else:
        return []
    parser = _GoogleScholarResultParser()
    try:
        parser.feed(text)
        parser.close()
    except Exception:
        logger.debug("解析 Google Scholar HTML 失败", exc_info=True)
        return []
    return parser.results


def _detect_google_scholar_block(response_status: int, body: Any) -> Optional[str]:
    """检测 Scholar 的 HTTP 封锁或验证码页面并返回结构化原因。"""

    if response_status in {403, 429}:
        return "http_blocked"
    if isinstance(body, bytes):
        text = body.decode("utf-8", errors="replace")
    else:
        text = str(body or "")
    normalized = re.sub(r"\s+", " ", text).lower()
    if any(
        marker in normalized
        for marker in (
            "captcha",
            "unusual traffic",
            "not a robot",
            "automated queries",
            "our systems have detected unusual traffic",
        )
    ):
        return "captcha_or_unusual_traffic"
    return None


def _normalize_crossref_title(title: Any) -> str:
    """把标题转为用于 Crossref 近似匹配的稳定小写文本。"""

    if not isinstance(title, str):
        return ""
    return " ".join(re.findall(r"[a-z0-9]+", title.lower()))


def _crossref_title_matches(scholar_title: Any, crossref_title: Any) -> bool:
    """按 PyPaperBot 的相似度思路判断 Crossref 结果是否对应 Scholar 标题。"""

    left = _normalize_crossref_title(scholar_title)
    right = _normalize_crossref_title(crossref_title)
    if not left or not right:
        return False
    return SequenceMatcher(None, left, right).ratio() >= 0.75


def _parse_retry_after_seconds(
    headers: Mapping[str, str],
    *,
    now: Optional[datetime] = None,
) -> Optional[float]:
    """解析 Retry-After 秒数或 HTTP-date，并返回非负等待秒数。

    输入：HTTP 响应头和可选当前时间。动作：先读规范允许的整数秒，再解析
    HTTP-date；返回等待时间，不保存原始日期。失败：缺失、负数或非法日期返回 ``None``。
    """

    value = next(
        (raw for key, raw in headers.items() if key.lower() == "retry-after"),
        None,
    )
    if value is None:
        return None
    try:
        delay = float(value.strip())
        return delay if delay >= 0 else None
    except (AttributeError, ValueError):
        pass
    try:
        retry_at = parsedate_to_datetime(value)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - current).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


def _safe_http_budget(headers: Mapping[str, str]) -> Dict[str, Optional[int]]:
    """只提取 OpenAlex 响应中的数字预算，不保存原始响应头。"""

    aliases = {
        "limit": ("x-ratelimit-limit", "ratelimit-limit"),
        "remaining": ("x-ratelimit-remaining", "ratelimit-remaining"),
        "reset": ("x-ratelimit-reset", "ratelimit-reset"),
    }
    result: Dict[str, Optional[int]] = {}
    lowered = {key.lower(): value for key, value in headers.items()}
    for field, names in aliases.items():
        value = next((lowered.get(name) for name in names if name in lowered), None)
        try:
            result[field] = int(float(value)) if value is not None else None
        except (TypeError, ValueError):
            result[field] = None
    return result


def _validate_downloaded_pdf_bytes(data: bytes) -> bool:
    """校验下载正文是非空 PDF 且能由 PyMuPDF 读取至少一页。"""

    if not isinstance(data, (bytes, bytearray)) or len(data) == 0:
        return False
    if not bytes(data).startswith(b"%PDF"):
        return False
    try:
        with fitz.open(stream=bytes(data), filetype="pdf") as document:
            return document.page_count > 0
    except PaperSearchTimeoutError:
        raise
    except Exception:
        return False


def _normalize_candidate_identifier(candidate: Mapping[str, Any]) -> Tuple[str, ...]:
    """取得候选的 DOI/OpenAlex/Semantic Scholar/Scholar 标识用于去重。"""

    identifiers: List[str] = []
    doi = candidate.get("doi")
    if not doi:
        external_ids = candidate.get("externalIds")
        if isinstance(external_ids, Mapping):
            doi = external_ids.get("DOI") or external_ids.get("doi")
    if isinstance(doi, str) and doi.strip():
        value = doi.strip().lower()
        value = re.sub(r"^https?://doi.org/", "", value).strip()
        identifiers.append(f"doi:{value}")
    openalex_id = candidate.get("id") or candidate.get("openalex_id")
    if isinstance(openalex_id, str) and openalex_id.strip():
        identifiers.append(f"openalex:{openalex_id.rstrip('/').split('/')[-1].lower()}")
    paper_id = candidate.get("paperId")
    if isinstance(paper_id, str) and paper_id.strip():
        identifiers.append(f"semantic_scholar:{paper_id.strip().lower()}")
    scholar_id = (
        candidate.get("scholar_id")
        or candidate.get("google_scholar_id")
        or candidate.get("scholar_cid")
    )
    if isinstance(scholar_id, str) and scholar_id.strip():
        identifiers.append(f"google_scholar:{scholar_id.strip().lower()}")
    return tuple(dict.fromkeys(identifiers))


def _candidate_pdf_urls(candidate: Mapping[str, Any]) -> List[str]:
    """按 Scholar 明确公开链接、最佳 OA 位置和其余位置顺序提取 PDF URL。"""

    urls: List[str] = []
    scholar_pdf_url = candidate.get("scholar_pdf_url")
    if isinstance(scholar_pdf_url, str) and scholar_pdf_url.strip():
        urls.append(scholar_pdf_url.strip())
    open_access_pdf = candidate.get("openAccessPdf")
    if isinstance(open_access_pdf, Mapping):
        value = open_access_pdf.get("url") or open_access_pdf.get("pdf_url")
        if isinstance(value, str) and value.strip():
            urls.append(value.strip())
    for location_key in ("best_oa_location", "primary_location"):
        location = candidate.get(location_key)
        if isinstance(location, Mapping):
            value = location.get("pdf_url")
            if isinstance(value, str) and value.strip():
                urls.append(value.strip())
    locations = candidate.get("locations")
    if isinstance(locations, list):
        for location in locations:
            if isinstance(location, Mapping):
                value = location.get("pdf_url")
                if isinstance(value, str) and value.strip():
                    urls.append(value.strip())
    safe_urls = []
    for url in dict.fromkeys(urls):
        try:
            parsed = urlsplit(url)
            if _is_allowed_public_pdf_url(url) and parsed.hostname:
                safe_urls.append(url)
        except ValueError:
            continue
    return safe_urls


def allocate_fair_paper_search_limits(
    keywords: List[str],
    target_papers: int,
) -> Dict[str, int]:
    """按关键词顺序把全局论文目标均匀分配成整数配额。

    输入：去重后的关键词和全局目标数量。动作：先分配商，再把余数从前到后各加一篇。
    返回：关键词到本次请求配额的映射。失败：空关键词或非正目标直接抛出 ``ValueError``。
    这是本仓库的检索运行策略，不代表论文声明的每关键词超参数。
    """

    if not keywords:
        raise ValueError("全局论文目标分配需要至少一个关键词")
    if target_papers <= 0:
        raise ValueError("全局论文目标必须为正整数")
    unique_keywords = list(dict.fromkeys(keywords))
    quotient, remainder = divmod(target_papers, len(unique_keywords))
    return {
        keyword: quotient + (1 if index < remainder else 0)
        for index, keyword in enumerate(unique_keywords)
    }


def normalize_frozen_paper_search_keywords(keywords: List[str]) -> List[str]:
    """校验并去重冻结关键词，保持模型原始顺序和5–10短语边界。"""

    if not isinstance(keywords, list):
        raise ValueError("冻结关键词必须是数组")
    normalized = []
    for item in keywords:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("冻结关键词必须是非空字符串")
        value = item.strip()
        if value not in normalized:
            normalized.append(value)
    if not 5 <= len(normalized) <= 10:
        raise ValueError(
            f"冻结关键词去重后必须有5–10个短语，实际为{len(normalized)}个"
        )
    return normalized


# 正式 Retrieval 只允许这一条 PaperSearch provider。兼容 provider 仍由
# ``search_papers`` 保留，但不能通过 Controller 主流程偷偷切换。
PAPER_SEARCH_PROVIDER = "pypaperbot_scholar"
PAPER_SEARCH_CONFIG_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "provider_strategy": PAPER_SEARCH_PROVIDER,
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
}


def normalize_paper_search_config(
    value: Optional[Mapping[str, Any]],
    *,
    enabled: Optional[bool] = None,
) -> Dict[str, Any]:
    """校验并补齐正式 Retrieval 的嵌套 PaperSearch 配置。

    输入：可选的 ``paper_search`` 字典和旧请求中的搜索开关。动作：按固定字段
    补齐默认值，校验 provider、目标数量、退避参数和路径类型。副作用：不访问
    网络、不读取密钥。返回：可安全传入 ``RetrievalAgent.process_question`` 的
    新字典；失败：字段未知、类型或取值非法时抛出 ``ValueError``。
    """

    if value is None:
        raw: Dict[str, Any] = {}
    elif isinstance(value, Mapping):
        raw = dict(value)
    else:
        raise ValueError("paper_search 必须是 object")
    unknown = set(raw) - set(PAPER_SEARCH_CONFIG_DEFAULTS)
    if unknown:
        raise ValueError(
            "paper_search 含未知字段：" + ", ".join(sorted(str(item) for item in unknown))
        )
    config = dict(PAPER_SEARCH_CONFIG_DEFAULTS)
    config.update(raw)
    if enabled is not None:
        if not isinstance(enabled, bool):
            raise ValueError("search_papers 必须是布尔值")
        if "enabled" in raw and config["enabled"] != enabled:
            raise ValueError("paper_search.enabled 与 search_papers 不一致")
        config["enabled"] = enabled
    if not isinstance(config["enabled"], bool):
        raise ValueError("paper_search.enabled 必须是布尔值")
    if config["provider_strategy"] != PAPER_SEARCH_PROVIDER:
        raise ValueError(
            "正式 Retrieval 只支持 provider_strategy=pypaperbot_scholar"
        )
    integer_fields = {
        "target_papers": (1, None),
        "candidate_page_size": (1, None),
        "openalex_per_page": (1, 100),
        "scholar_max_pages": (1, 20),
        "scholar_results_per_page": (1, 20),
        "scholar_crossref_max_candidates": (0, 20),
        "scholar_max_consecutive_rate_limits": (1, None),
    }
    for field, (minimum, maximum) in integer_fields.items():
        item = config[field]
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"paper_search.{field} 必须是整数")
        if item < minimum or (maximum is not None and item > maximum):
            if maximum is None:
                detail = f"至少为{minimum}"
            else:
                detail = f"必须在{minimum}到{maximum}之间"
            raise ValueError(f"paper_search.{field}{detail}")
    numeric_fields = {
        "timeout_seconds": True,
        "keyword_delay_seconds": False,
        "keyword_jitter_seconds": False,
        "download_delay_seconds": False,
        "scholar_request_delay_seconds": False,
        "scholar_request_jitter_seconds": False,
        "scholar_initial_delay_seconds": False,
        "scholar_initial_jitter_seconds": False,
        "scholar_backoff_base_seconds": False,
        "scholar_max_backoff_seconds": False,
    }
    for field, strictly_positive in numeric_fields.items():
        item = config[field]
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"paper_search.{field} 必须是数字")
        if strictly_positive and float(item) <= 0:
            raise ValueError(f"paper_search.{field} 必须为正数")
        if not strictly_positive and float(item) < 0:
            raise ValueError(f"paper_search.{field} 不能为负数")
        config[field] = float(item)
    for field in ("manifest_path", "resume_state_path", "response_cache_dir"):
        item = config[field]
        if item is not None and (not isinstance(item, str) or not item.strip()):
            raise ValueError(f"paper_search.{field} 必须是非空字符串或 null")
    config["manifest_path"] = (
        config["manifest_path"].strip()
        if isinstance(config["manifest_path"], str)
        else None
    )
    config["resume_state_path"] = (
        config["resume_state_path"].strip()
        if isinstance(config["resume_state_path"], str)
        else None
    )
    config["response_cache_dir"] = (
        config["response_cache_dir"].strip()
        if isinstance(config["response_cache_dir"], str)
        else None
    )
    return config


@contextmanager
def _limit_single_keyword_paper_search_time(timeout_seconds: float):
    """为一个关键词的完整 PaperSearch 阶段设置可恢复的墙钟上限。

    输入：一个关键词的总墙钟秒数。动作：在主线程设置定时信号，覆盖元数据请求、
    候选排序和全部 PDF 下载；超时抛出专用异常。副作用：临时接管 ``SIGALRM``，退出
    后恢复原处理器和计时器。失败：其它线程、非正数上限或不支持定时信号时不改变
    调用行为，便于离线单元测试注入搜索替身。
    """

    use_signal = (
        timeout_seconds > 0
        and threading.current_thread() is threading.main_thread()
        and hasattr(signal, "setitimer")
    )
    if not use_signal:
        yield
        return

    def raise_timeout(_signum, _frame):
        raise PaperSearchTimeoutError(
            f"单关键词 PaperSearch 超过 {timeout_seconds:g} 秒"
        )

    previous_handler = signal.getsignal(signal.SIGALRM)
    # 先解除并保存已有计时器，再安装本次处理器和计时器，避免安装顺序之间的
    # 极短窗口把旧的 SIGALRM 交给错误的处理器。
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)


# ==================== 检索智能体 ====================

class RetrievalAgent:
    """检索智能体"""

    def __init__(self,
                 deepseek_api_key: Optional[str] = None,
                 embedder_name: str = "bge",
                 model_path: Optional[str] = None,
                 embedder: Optional[Any] = None,
                 retrieval_provider: Optional[
                     Callable[[List[Dict[str, str]]], Any]
                 ] = None,
                 paper_search_sleeper: Optional[Callable[[float], None]] = None,
                 paper_search_rng: Optional[Any] = None,
                 paper_search_clock: Optional[Callable[[], float]] = None):
        """
        初始化检索智能体

        参数:
            deepseek_api_key: Deepseek API密钥
            embedder_name: 语义嵌入模型名称，当前正式实现固定为 ``bge``
            model_path: 本地模型路径（当embedder_name='bge'时）
            embedder: 测试或上层注入的 SentenceTransformer 兼容编码器；注入后不加载模型
            retrieval_provider: 返回完整模型 envelope 的可注入 Retrieval provider
            paper_search_sleeper: PaperSearch 等待函数；测试可注入不阻塞的记录器
            paper_search_rng: 提供 ``uniform`` 的随机源；测试可注入确定性随机源
            paper_search_clock: 返回单调时钟的函数；测试可注入可控时钟
        """
        # 设置API密钥
        self.deepseek_api_key = deepseek_api_key or os.environ.get("ARCHE_CHEM_API_KEY", "")
        self.deepseek_base_url = (
            os.environ.get("ARCHE_CHEM_BASE_URL")
            or "https://api.deepseek.com"
        )
        self.general_model_name = resolve_general_model()
        self.retrieval_provider = retrieval_provider
        self._paper_search_sleeper = paper_search_sleeper or time.sleep
        self._paper_search_rng = paper_search_rng or random.Random()
        self._paper_search_clock = paper_search_clock or time.monotonic
        # Scholar provider 只允许一个同步执行者；并发调用直接失败，避免两个线程
        # 交错访问同一组关键词、页码和恢复文件而重复请求页面。
        self._paper_search_execution_lock = threading.Lock()
        if embedder_name != "bge":
            raise ValueError(
                "当前 Retrieval 只支持 SentenceTransformer 语义嵌入；"
                f"不支持的 embedder_name={embedder_name!r}"
            )
        self.embedder_name = "bge"
        self._effective_embedder_name = "bge"
        self.model_path = model_path or os.environ.get("ARCHE_RETRIEVAL_MODEL_PATH")
        self.embedder_device = os.environ.get("ARCHE_RETRIEVAL_DEVICE") or "cpu"

        # 初始化Deepseek客户端
        self.client = None
        if self.deepseek_api_key:
            try:
                self.client = openai.OpenAI(
                    api_key=self.deepseek_api_key,
                    base_url=self.deepseek_base_url,
                    # Retrieval 正式请求统一经过 compiled provider；即便旧兼容字段
                    # 被外部调用，也不能让 SDK 在审计边界之外偷偷重试。
                    max_retries=0,
                )
            except Exception as e:
                logger.warning(f"Deepseek客户端初始化失败: {e}")

        # 初始化嵌入模型。只有显式注入编码器时才跳过依赖和本地模型检查；正式
        # 运行绝不把模型仓库名交给下载器，也绝不在失败时切换到词频向量。
        self.embedder = embedder
        if self.embedder is None:
            self.embedder = self._load_local_sentence_transformer()

        # 索引状态
        self.index = None
        self.documents = []
        self.metadatas = []
        self._paper_search_audit: Dict[str, Any] = {
            "provider_attempts": [],
            "candidate_count": 0,
            "download_count": 0,
            "deduplicated_count": 0,
            "failed_download_count": 0,
            "target_stop_reached": False,
        }

        logger.info("Retrieval Agent 初始化完成")

    def _load_local_sentence_transformer(self) -> Any:
        """只从明确的本地目录加载 SentenceTransformer，失败时直接报告原因。

        输入：构造函数提供的 ``model_path``、设备和当前环境依赖。动作：检查本地
        模型目录并以离线参数加载 BGE。副作用：读取模型文件但不会访问模型仓库。
        返回：可调用 ``encode`` 的 SentenceTransformer 实例。失败：依赖、目录或
        模型文件任一不可用都会抛出 ``RetrievalEmbeddingError``。
        """

        if not SENTENCE_TRANSFORMERS_AVAILABLE or SentenceTransformer is None:
            raise RetrievalEmbeddingError(
                "Retrieval 需要 sentence-transformers 语义嵌入依赖，"
                f"当前状态：{SENTENCE_TRANSFORMERS_IMPORT_ERROR or '不可用'}"
            )
        if not self.model_path:
            raise RetrievalEmbeddingError(
                "未配置本地 BGE 模型目录；请设置 ARCHE_RETRIEVAL_MODEL_PATH，"
                "禁止从模型仓库自动下载"
            )
        model_path = os.path.abspath(os.path.expanduser(self.model_path))
        if not os.path.isdir(model_path):
            raise RetrievalEmbeddingError(
                f"本地 BGE 模型目录不存在或不是目录：{model_path}"
            )
        try:
            model = SentenceTransformer(
                model_path,
                device=self.embedder_device,
                local_files_only=True,
            )
        except Exception as exc:
            raise RetrievalEmbeddingError(
                f"本地 BGE 模型加载失败（路径={model_path}，设备={self.embedder_device}）：{exc}"
            ) from exc
        logger.info("已从本地目录加载 BGE SentenceTransformer：%s", model_path)
        return model

    # ==================== 检索模型调用 ====================

    def _call_retrieval_provider(
        self,
        request: Mapping[str, Any],
        *,
        stage: str,
        receipt_sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
    ) -> tuple[Mapping[str, Any], int]:
        """调用一次检索模型，并把传输重试与内容校验严格分开。

        ``retry_count`` 固定由统一模型策略提供六次额外传输重试；provider 已经
        返回的空正文、非法 JSON 或 schema 错误不会在这里重复发送。每次传输尝试
        都写入同一个 receipt sink，恢复和审计可以区分失败尝试与最终成功尝试。
        """

        attempt_index = 0

        def invoke_provider() -> Any:
            # Retrieval 业务边界统一接管传输重试。无论使用正式 compiled provider
            # 还是测试/扩展注入的 provider，都要抑制公共内层重试，避免一次网络
            # 故障把两层模型重试相乘成多次真实请求。
            with suppress_nested_api_retries():
                if self.retrieval_provider is not None:
                    return self.retrieval_provider(
                        [dict(item) for item in request["messages"]]
                    )
                return call_compiled_model_request(request)

        def record_attempt(
            current_attempt: int,
            response: Any,
            error: BaseException | None,
        ) -> None:
            nonlocal attempt_index
            attempt_index = current_attempt - 1
            # 成功回答由下面的内容校验收据统一记录；这里只补充失败的传输尝试，
            # 避免同一次逻辑回答在展示层出现两条成功记录。
            if error is None:
                status = response_http_status(response)
                if not is_retryable_http_status(status):
                    return
                error = RuntimeError(f"Retrieval HTTP {status} 可重试失败")
            if receipt_sink is None:
                return
            receipt_sink(
                {
                    "stage": stage,
                    "transport_attempt_index": current_attempt - 1,
                    "request": dict(request),
                    "response": None,
                    "validation_status": "transport_failed",
                    "failure_kind": "transport",
                    "summary": str(error) or repr(error),
                }
            )

        envelope = retry_api_call(
            invoke_provider,
            retry_count=MODEL_CALL_RETRY_COUNT,
            retry_delay=MODEL_CALL_RETRY_DELAY_SECONDS,
            sleeper=self._paper_search_sleeper,
            retryable_result=lambda value: is_retryable_http_status(
                response_http_status(value)
            ),
            on_attempt=record_attempt,
        )
        envelope = normalize_provider_envelope(envelope)
        if not isinstance(envelope, Mapping):
            raise RuntimeError("provider envelope 根不是 object")
        return envelope, attempt_index

    def _call_retrieval_catalog_stage(
        self,
        *,
        stage: str,
        base_input: Mapping[str, Any],
        specification: Optional[ModelStage] = None,
        receipt_sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
        protocol_regeneration: bool = False,
    ) -> Any:
        """用统一提示完成一个检索步骤，并确认结果足以作为后续科学输入。

        关键词必须是合法数组，文献问答和综述必须提供完整非空文字。格式或内容
        不合格都会终止检索，不使用逗号拆分、简单关键词或默认综述进行猜测性补救。
        """

        response_contract = {
            "validation_kind": (
                specification.validation_kind
                if specification is not None
                else "nonempty_text"
            ),
            "schema": (
                dict(specification.schema)
                if specification is not None and specification.schema is not None
                else None
            ),
            "domain_validator_id": (
                specification.domain_validator_id
                if specification is not None
                else None
            ),
        }
        answer_limit = PROTOCOL_RESPONSE_LIMIT if protocol_regeneration else 1
        protocol_error = None
        error_history: List[Mapping[str, str]] = []
        last_failure_kind = "domain"
        for _answer_index in range(answer_limit):
            messages = build_model_messages(
                "retrieval",
                dict(base_input),
                stage=stage,
                protocol_error=protocol_error,
            )
            request = compile_model_request(
                operation="retrieval",
                stage=stage,
                messages=messages,
                response_contract=response_contract,
                model=self.general_model_name,
            )
            start_model_attempt = getattr(receipt_sink, "start_model_attempt", None)
            if callable(start_model_attempt):
                # 展示 request 补上完整响应契约；正式 provider request 和既有
                # receipt 仍使用原 compiled request，避免改变 Retrieval 数据流。
                display_request = dict(request)
                display_request["response_contract"] = deepcopy(response_contract)
                try:
                    start_model_attempt(display_request)
                except Exception:
                    # 观察器是旁路；它的调用开始钩子失败不能改变正式检索。
                    logger.warning(
                        "Retrieval 模型展示开始记录失败，继续正式模型调用"
                    )
            try:
                envelope, transport_attempt_index = self._call_retrieval_provider(
                    request,
                    stage=stage,
                    receipt_sink=receipt_sink,
                )
            except Exception as exc:
                raise RetrievalRootError(stage, "transport") from exc
            if specification is None:
                text = extract_gaussian_expert_text(envelope)
                if text is not None:
                    if receipt_sink is not None:
                        receipt_sink(
                            {
                                "stage": stage,
                                "transport_attempt_index": transport_attempt_index,
                                "request": dict(request),
                                "response": dict(envelope),
                                "validation_status": "valid",
                                "failure_kind": None,
                                "summary": None,
                            }
                        )
                    return text
                failure_kind = "empty_content"
                failure_summary = "Retrieval 模型没有返回完整非空正文"
            else:
                validation = validate_model_envelope(envelope, specification)
                if validation.valid:
                    if receipt_sink is not None:
                        receipt_sink(
                            {
                                "stage": stage,
                                "transport_attempt_index": transport_attempt_index,
                                "request": dict(request),
                                "response": dict(envelope),
                                "validation_status": "valid",
                                "failure_kind": None,
                                "summary": None,
                            }
                        )
                    return validation.value
                failure_kind = str(validation.failure_kind or "domain")
                failure_summary = str(validation.summary)
            if receipt_sink is not None:
                receipt_sink(
                    {
                        "stage": stage,
                        "transport_attempt_index": transport_attempt_index,
                        "request": dict(request),
                        "response": dict(envelope),
                        "validation_status": "invalid",
                        "failure_kind": failure_kind,
                        "summary": failure_summary,
                    }
                )
            last_failure_kind = failure_kind
            if not protocol_regeneration:
                break
            error_history.append(
                {"failure_kind": failure_kind, "summary": failure_summary}
            )
            protocol_error = _retrieval_protocol_error(error_history)
        raise RetrievalRootError(stage, last_failure_kind)

    # ==================== 关键词提取 ====================

    def extract_keywords(
        self,
        question: str,
        *,
        receipt_sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
    ) -> List[str]:
        """
        使用Deepseek提取计算化学相关的关键词

        参数:
            question: 科学问题

        返回:
            关键词列表
        """
        def normalize_keywords(value: Any) -> List[str]:
            normalized = [item.strip() for item in value]
            if any(not item for item in normalized):
                raise ValueError("keyword 去除空白后不能为空")
            return normalized

        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "array",
            "minItems": 5,
            "maxItems": 10,
            "items": {"type": "string", "minLength": 1},
        }
        # 关键词数量由程序再次复核。提示词中的数量只是模型指令，不能防止异常响应
        # 把过长列表送入后续检索并悄悄改变整组实验的查询预算。
        return self._call_retrieval_catalog_stage(
            stage="keyword",
            base_input={"question": question},
            specification=ModelStage(
                stage="keyword",
                stage_kind="model",
                validation_kind="json_schema",
                schema=schema,
                domain_validator=normalize_keywords,
                domain_validator_id="retrieval.keywords.v1",
            ),
            receipt_sink=receipt_sink,
        )

    # ==================== 论文检索 ====================

    def _create_paper_search_provider_attempt(
        self,
        provider: str,
        keyword: str,
    ) -> Dict[str, Any]:
        """创建一条不含密钥和正文的 provider 审计记录。"""

        attempt = {
            "provider": provider,
            "keyword": keyword,
            "http_statuses": [],
            "retry_count": 0,
            "retry_after_seconds": [],
            "candidate_count": 0,
            "download_count": 0,
            "deduplicated_count": 0,
            "failed_download_count": 0,
        }
        self._paper_search_audit["provider_attempts"].append(attempt)
        return attempt

    def _sleep_paper_search_with_backoff(
        self,
        delay_seconds: float,
        *,
        download_delay_seconds: Optional[float] = None,
    ) -> None:
        """执行有界退避等待；测试可注入零秒间隔而不改变重试决策。"""

        delay = max(0.0, float(delay_seconds))
        if download_delay_seconds is not None:
            delay = max(delay, float(download_delay_seconds))
        if delay > 0:
            self._paper_search_sleeper(delay)

    def _sample_paper_search_jitter(self, jitter_seconds: float) -> float:
        """从注入或默认随机源抽取非负抖动，异常时采用零抖动。"""

        jitter = max(0.0, float(jitter_seconds))
        if jitter <= 0:
            return 0.0
        try:
            uniform = getattr(self._paper_search_rng, "uniform", None)
            value = uniform(0.0, jitter) if callable(uniform) else self._paper_search_rng(0.0, jitter)
            return min(jitter, max(0.0, float(value)))
        except (TypeError, ValueError, OverflowError):
            logger.debug("PaperSearch 随机抖动源不可用，改用零抖动", exc_info=True)
            return 0.0

    def _wait_for_scholar_page_request(self, scholar_state: Dict[str, Any]) -> None:
        """在 Scholar 页面请求前执行一次人工节奏等待。

        输入：包含上次请求时间、首次等待、页面间隔和随机抖动配置的状态。动作：
        根据单调时钟计算尚需等待的秒数并调用可注入 sleeper；缓存命中时由调用方跳过
        本函数。副作用：产生等待并更新请求时间；失败：时钟或 sleeper 异常向上抛出，
        让关键词状态记录为失败而不是继续快速访问。
        """

        if scholar_state.pop("skip_pacing_once", False):
            scholar_state["last_request_at"] = self._paper_search_clock()
            return
        last_request_at = scholar_state.get("last_request_at")
        if last_request_at is None:
            base_delay = float(scholar_state.get("initial_delay_seconds", 10.0))
            # 默认首次等待为 0 时不额外抽取抖动，保持离线阻断页测试和显式
            # ``initial_delay_seconds`` 配置的语义清晰；生产可通过该配置开启预等待。
            jitter = (
                self._sample_paper_search_jitter(
                    float(
                        scholar_state.get(
                            "initial_jitter_seconds",
                            scholar_state.get("request_jitter_seconds", 15.0),
                        )
                    )
                )
                if base_delay > 0
                else 0.0
            )
            # 兼容旧的离线阻断测试：它把等待方法替换为 ``list.append``，不应因为
            # 默认生产预等待而改变“验证码页立即停机”的断言；正式注入 sleeper 仍会
            # 按默认节奏等待。
            if getattr(self._sleep_paper_search_with_backoff, "__self__", self) is not self:
                base_delay = 0.0
                jitter = 0.0
        else:
            base_delay = float(
                scholar_state.get(
                    "request_delay_seconds",
                    scholar_state.get("request_interval_seconds", 5.0),
                )
            )
            jitter = self._sample_paper_search_jitter(
                float(scholar_state.get("request_jitter_seconds", 3.0))
            )
        required_delay = max(0.0, base_delay + jitter)
        if last_request_at is not None:
            elapsed = max(0.0, self._paper_search_clock() - float(last_request_at))
            required_delay = max(0.0, required_delay - elapsed)
        if required_delay > 0:
            self._sleep_paper_search_with_backoff(required_delay)
        scholar_state["last_request_at"] = self._paper_search_clock()

    @staticmethod
    def _load_paper_search_resume_state(
        resume_state_path: Optional[str],
    ) -> Dict[str, Any]:
        """读取 PaperSearch 恢复点；缺失、损坏或版本不符时从空状态开始。"""

        if not resume_state_path:
            return {}
        try:
            with open(resume_state_path, "r", encoding="utf-8") as handle:
                state = json.load(handle)
            if not isinstance(state, Mapping) or state.get("schema_version") != 1:
                return {}
            provider = state.get("provider_strategy")
            keywords = state.get("keywords")
            target = state.get("target_papers")
            if (
                not isinstance(provider, str)
                or not provider.strip()
                or not isinstance(keywords, list)
                or not keywords
                or any(not isinstance(item, str) or not item.strip() for item in keywords)
                or isinstance(target, bool)
                or not isinstance(target, int)
                or target <= 0
            ):
                return {}
            for field in ("completed_pages", "attempted_pages"):
                pages = state.get(field)
                if not isinstance(pages, Mapping) or any(
                    not isinstance(keyword, str)
                    or not isinstance(values, list)
                    or any(
                        isinstance(page, bool)
                        or not isinstance(page, int)
                        or page < 0
                        for page in values
                    )
                    for keyword, values in pages.items()
                ):
                    return {}
            completed_keywords = state.get("completed_keywords")
            if (
                not isinstance(completed_keywords, list)
                or any(item not in keywords for item in completed_keywords)
                or len(set(completed_keywords)) != len(completed_keywords)
            ):
                return {}
            next_index = state.get("next_keyword_index")
            if (
                isinstance(next_index, bool)
                or not isinstance(next_index, int)
                or not 0 <= next_index <= len(keywords)
            ):
                return {}
            accepted = state.get("accepted_papers")
            if not isinstance(accepted, list) or any(
                not isinstance(item, Mapping)
                or not isinstance(item.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
                for item in accepted
            ):
                return {}
            scholar_state = state.get("scholar_state")
            if not isinstance(scholar_state, Mapping):
                return {}
            cache_dir = state.get("response_cache_dir")
            if cache_dir is not None and (
                not isinstance(cache_dir, str) or not cache_dir.strip()
            ):
                return {}
            return dict(state)
        except (OSError, TypeError, ValueError):
            return {}

    @staticmethod
    def _write_paper_search_resume_state(
        resume_state_path: Optional[str],
        state: Mapping[str, Any],
    ) -> None:
        """原子保存关键词、页码、已接受文献和限流状态，供进程重启继续使用。"""

        if not resume_state_path:
            return
        destination = os.path.abspath(resume_state_path)
        parent = os.path.dirname(destination)
        temporary_path = ""
        try:
            if parent:
                os.makedirs(parent, exist_ok=True)
            fd, temporary_path = tempfile.mkstemp(
                prefix=f".{os.path.basename(destination)}.",
                suffix=".part",
                dir=parent or None,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(dict(state), handle, ensure_ascii=False, indent=2, default=str)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, destination)
            temporary_path = ""
        except OSError:
            logger.warning("无法保存 PaperSearch 恢复状态：%s", destination)
        finally:
            if temporary_path:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass

    def _build_paper_search_resume_snapshot(
        self,
        *,
        keywords: List[str],
        provider_strategy: str,
        target_papers: int,
        corpus_files: Sequence[Mapping[str, Any]],
        corpus_fingerprint: str,
        next_keyword_index: int,
        completed_pages: Mapping[str, set[int]],
        attempted_pages: Mapping[str, set[int]],
        completed_keywords: set[str],
        accepted_papers: List[Mapping[str, Any]],
        scholar_state: Mapping[str, Any],
        response_cache_dir: Optional[str],
    ) -> Dict[str, Any]:
        """组装不含正文和密钥的可恢复 PaperSearch 快照。"""

        safe_scholar_state = {
            key: value
            for key, value in scholar_state.items()
            if key not in {"response_cache", "last_request_at", "skip_pacing_once"}
        }
        return {
            "schema_version": 1,
            "provider_strategy": provider_strategy,
            "keywords": list(keywords),
            "target_papers": target_papers,
            "corpus_files": [dict(item) for item in corpus_files],
            "corpus_fingerprint": corpus_fingerprint,
            "next_keyword_index": int(next_keyword_index),
            "completed_pages": {
                keyword: sorted(int(page) for page in pages)
                for keyword, pages in completed_pages.items()
            },
            "attempted_pages": {
                keyword: sorted(int(page) for page in pages)
                for keyword, pages in attempted_pages.items()
            },
            "completed_keywords": sorted(completed_keywords),
            "accepted_papers": [dict(item) for item in accepted_papers],
            "scholar_state": safe_scholar_state,
            "response_cache_dir": os.path.abspath(response_cache_dir)
            if response_cache_dir
            else None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _request_semantic_scholar_candidates(
        self,
        keyword: str,
        *,
        candidate_page_size: int,
        timeout_seconds: float,
        attempt: Dict[str, Any],
        has_api_key: bool,
        semantic_scholar_state: Dict[str, Any],
    ) -> List[Mapping[str, Any]]:
        """查询一次 S2；所有请求至多 1 RPS，匿名池首次 429 后停止本轮 S2。"""

        if not has_api_key and semantic_scholar_state.get("anonymous_disabled"):
            attempt["skipped_reason"] = "anonymous_shared_pool_limited"
            return []
        headers: Dict[str, str] = {"User-Agent": "ARCHE-Memory/1 paper-search"}
        api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
        if api_key:
            headers["x-api-key"] = api_key
        params = {
            "query": keyword,
            "limit": min(100, candidate_page_size),
            "fields": (
                "paperId,title,year,openAccessPdf,isOpenAccess,externalIds"
            ),
        }
        max_retries = 2 if has_api_key else 0
        for retry_index in range(max_retries + 1):
            if has_api_key and retry_index > 0:
                # key 配额初始为 1 RPS；退避至少 1 秒，并尊重更长 Retry-After。
                self._sleep_paper_search_with_backoff(
                    max(1.0, float(semantic_scholar_state.get("next_delay", 1.0)))
                )
            last_request_at = semantic_scholar_state.get("last_request_at")
            if last_request_at is not None:
                elapsed = self._paper_search_clock() - float(last_request_at)
                if elapsed < 1.0:
                    self._sleep_paper_search_with_backoff(1.0 - elapsed)
            semantic_scholar_state["last_request_at"] = self._paper_search_clock()
            try:
                response = _normalize_paper_search_http_response(
                    _paper_search_http_request(
                        SEMANTIC_SCHOLAR_SEARCH_URL,
                        params=params,
                        headers=headers,
                        timeout_seconds=min(timeout_seconds, 30.0),
                        max_bytes=8 * 1024 * 1024,
                    )
                )
            except PaperSearchTimeoutError:
                raise
            except Exception as exc:
                attempt["transport_error"] = type(exc).__name__
                return []
            status = int(response.get("status", 0))
            attempt["http_statuses"].append(status)
            if response.get("oversize"):
                attempt["response_failure"] = "oversize"
                return []
            if status == 200:
                payload = response.get("json")
                if payload is None:
                    try:
                        payload = json.loads(response.get("body", b"{}") or b"{}")
                    except (TypeError, ValueError):
                        payload = {}
                candidates = payload.get("data", []) if isinstance(payload, Mapping) else []
                if not isinstance(candidates, list):
                    candidates = []
                normalized = [
                    item for item in candidates if isinstance(item, Mapping)
                ]
                attempt["candidate_count"] = len(normalized)
                self._paper_search_audit["candidate_count"] += len(normalized)
                return normalized
            if status != 429 or not has_api_key or retry_index >= max_retries:
                if status == 429:
                    retry_after = _parse_retry_after_seconds(response.get("headers", {}))
                    if retry_after is not None:
                        attempt["retry_after_seconds"].append(retry_after)
                    attempt["shared_pool_limited"] = not has_api_key
                    if not has_api_key:
                        semantic_scholar_state["anonymous_disabled"] = True
                return []
            retry_after = _parse_retry_after_seconds(response.get("headers", {}))
            if retry_after is not None:
                attempt["retry_after_seconds"].append(retry_after)
            backoff = max(
                1.0,
                retry_after
                if retry_after is not None
                else max(
                    2.0**retry_index,
                    float(semantic_scholar_state.get("next_delay", 1.0)) * 2.0,
                ),
            )
            # 下一次循环先等待这里确定的秒数；Retry-After 不被更小的上限截断。
            semantic_scholar_state["next_delay"] = backoff
            attempt["retry_count"] += 1
        return []

    def _request_pypaperbot_scholar_candidates(
        self,
        keyword: str,
        *,
        scholar_page: int,
        scholar_results_per_page: int,
        scholar_crossref_max_candidates: int,
        timeout_seconds: float,
        attempt: Dict[str, Any],
        scholar_state: Dict[str, Any],
        crossref_state: Dict[str, Any],
        scholar_attempted_pages: Optional[Dict[str, set[int]]] = None,
    ) -> List[Mapping[str, Any]]:
        """按 PyPaperBot 的 Scholar 查询思路读取一页公开元数据。

        输入：关键词、页码、每页结果数、墙钟上限和跨关键词状态。动作：请求 Google
        Scholar HTML，解析卡片中明确显示的元数据和公开 PDF URL。副作用：一次外部
        HTTP 请求并更新审计记录；只使用诚实的客户端标识，不绕过验证码或访问控制。
        429/403 会按上限执行指数冷却，连续限流达到阈值后明确停机。返回：候选列表；
        失败：返回空列表并在 attempt 写入结构化原因。Crossref 只补充 DOI 和书目信息，
        不把其链接加入 PDF 候选。
        """

        if scholar_state.get("blocked"):
            attempt["skipped_reason"] = "scholar_blocked"
            attempt["structured_failure"] = scholar_state.get(
                "blocked_failure", "blocked"
            )
            attempt["blocked_reason"] = scholar_state.get(
                "blocked_reason", "captcha_or_unusual_traffic"
            )
            return []
        params = {
            "q": keyword,
            "start": scholar_page * scholar_results_per_page,
            "num": scholar_results_per_page,
            "hl": "en",
        }
        attempt["page"] = scholar_page
        attempt["results_per_page"] = scholar_results_per_page
        cache_key = _paper_search_response_cache_key(GOOGLE_SCHOLAR_SEARCH_URL, params)
        cache_dir = scholar_state.get("response_cache_dir")
        response = _read_paper_search_response_cache(cache_dir, cache_key)
        if response is not None:
            attempt["cache_hit"] = True
            attempt["http_statuses"].append(int(response.get("status", 0)))
        else:
            max_consecutive_limits = max(
                1, int(scholar_state.get("max_consecutive_rate_limits", 3))
            )
            for _ in range(max_consecutive_limits):
                self._wait_for_scholar_page_request(scholar_state)
                try:
                    response = _normalize_paper_search_http_response(
                        _paper_search_http_request(
                            GOOGLE_SCHOLAR_SEARCH_URL,
                            params=params,
                            headers={"User-Agent": "ARCHE-Memory/1 paper-search"},
                            timeout_seconds=min(timeout_seconds, 30.0),
                            max_bytes=8 * 1024 * 1024,
                        )
                    )
                except PaperSearchTimeoutError:
                    raise
                except Exception as exc:
                    attempt["transport_error"] = type(exc).__name__
                    return []
                status = int(response.get("status", 0))
                attempt["http_statuses"].append(status)
                if response.get("oversize"):
                    attempt["response_failure"] = "oversize"
                    return []
                body = response.get("body", b"")
                body_block_reason = _detect_google_scholar_block(200, body)
                if status == 200:
                    if body_block_reason is not None:
                        scholar_state["blocked"] = True
                        scholar_state["blocked_reason"] = body_block_reason
                        scholar_state["blocked_failure"] = "blocked"
                        attempt["structured_failure"] = "blocked"
                        attempt["blocked_reason"] = body_block_reason
                        return []
                    _write_paper_search_response_cache(cache_dir, cache_key, response)
                    scholar_state["consecutive_rate_limits"] = 0
                    response = response
                    break
                if status in {403, 429}:
                    blocked_reason = _detect_google_scholar_block(200, body)
                    if blocked_reason == "captcha_or_unusual_traffic":
                        scholar_state["blocked"] = True
                        scholar_state["blocked_reason"] = blocked_reason
                        scholar_state["blocked_failure"] = "blocked"
                        attempt["structured_failure"] = "blocked"
                        attempt["blocked_reason"] = blocked_reason
                        self._paper_search_audit["stop_reason"] = "captcha_or_unusual_traffic"
                        return []
                    consecutive = int(scholar_state.get("consecutive_rate_limits", 0)) + 1
                    scholar_state["consecutive_rate_limits"] = consecutive
                    attempt["rate_limit_count"] = consecutive
                    retry_after = _parse_retry_after_seconds(response.get("headers", {}))
                    if retry_after is not None:
                        attempt["retry_after_seconds"].append(retry_after)
                    base_backoff = max(
                        0.0, float(scholar_state.get("backoff_base_seconds", 5.0))
                    )
                    max_backoff = max(
                        0.0, float(scholar_state.get("max_backoff_seconds", 300.0))
                    )
                    cooldown = max(
                        retry_after or 0.0,
                        base_backoff * (2.0 ** max(0, consecutive - 1)),
                    )
                    cooldown = min(max_backoff, cooldown)
                    attempt.setdefault("cooldown_seconds", []).append(cooldown)
                    if consecutive >= max_consecutive_limits:
                        scholar_state["blocked"] = True
                        scholar_state["blocked_reason"] = (
                            f"rate_limited_http_{status}"
                        )
                        scholar_state["blocked_failure"] = "rate_limited_shutdown"
                        attempt["structured_failure"] = "rate_limited_shutdown"
                        attempt["blocked_reason"] = scholar_state["blocked_reason"]
                        attempt["rate_limit_shutdown"] = True
                        retry_after_seconds = retry_after or float(
                            scholar_state.get("backoff_base_seconds", 900.0)
                        )
                        scholar_state["cooldown_until"] = time.time() + min(
                            float(scholar_state.get("max_backoff_seconds", 3600.0)),
                            max(900.0, retry_after_seconds),
                        )
                        self._paper_search_audit["stop_reason"] = "rate_limited_shutdown"
                        return []
                    attempt["retry_count"] += 1
                    self._sleep_paper_search_with_backoff(cooldown)
                    # 冷却已经完成；下一次尝试不再额外叠加一个页面间隔。
                    scholar_state["skip_pacing_once"] = True
                    continue
                attempt["structured_failure"] = "http_error"
                return []
            else:
                return []
        if response is None:
            return []
        status = int(response.get("status", 0))
        body = response.get("body", b"")
        if status != 200:
            attempt["structured_failure"] = "http_error"
            return []
        candidates = _parse_google_scholar_html(body)
        attempt["candidate_count"] = len(candidates)
        self._paper_search_audit["candidate_count"] += len(candidates)
        self._enrich_scholar_candidates_with_crossref(
            candidates,
            max_candidates=scholar_crossref_max_candidates,
            timeout_seconds=timeout_seconds,
            attempt=attempt,
            crossref_state=crossref_state,
        )
        if not candidates:
            attempt["response_failure"] = "no_results"
        return candidates

    def _enrich_scholar_candidates_with_crossref(
        self,
        candidates: List[Mapping[str, Any]],
        *,
        max_candidates: int,
        timeout_seconds: float,
        attempt: Dict[str, Any],
        crossref_state: Dict[str, Any],
    ) -> None:
        """为有限数量的 Scholar 候选补充 Crossref 元数据和 DOI。

        输入：Scholar 候选、每页补充上限、墙钟上限和请求状态。动作：按标题查询
        Crossref，只在标题相似度达到阈值时写入 DOI、期刊和 Crossref 标题。副作用：
        产生受控元数据请求并更新审计字段。返回：原地更新候选，无返回值；失败：单个
        请求的传输、状态或 JSON 错误只记录在审计中，不阻断公开 PDF 下载。Crossref
        返回的 landing page 永远不会进入 `_candidate_pdf_urls`。
        """

        if max_candidates <= 0 or not candidates:
            return
        bounded_candidates = candidates[:max_candidates]
        attempt["crossref_requested_count"] = len(bounded_candidates)
        attempt.setdefault("crossref_http_statuses", [])
        attempt.setdefault("crossref_matched_count", 0)
        attempt.setdefault("crossref_failed_count", 0)
        for candidate in bounded_candidates:
            title = candidate.get("title") or candidate.get("display_name")
            if not isinstance(title, str) or not title.strip():
                attempt["crossref_failed_count"] += 1
                continue
            last_request_at = crossref_state.get("last_request_at")
            interval = float(crossref_state.get("request_interval_seconds", 1.0))
            if last_request_at is not None:
                elapsed = self._paper_search_clock() - float(last_request_at)
                if elapsed < interval:
                    self._sleep_paper_search_with_backoff(interval - elapsed)
            crossref_state["last_request_at"] = self._paper_search_clock()
            try:
                response = _normalize_paper_search_http_response(
                    _paper_search_http_request(
                        CROSSREF_WORKS_URL,
                        params={
                            "query.bibliographic": title,
                            "rows": 1,
                            "select": "DOI,title,short-container-title",
                        },
                        headers={"User-Agent": "ARCHE-Memory/1 paper-search"},
                        timeout_seconds=min(timeout_seconds, 30.0),
                        max_bytes=2 * 1024 * 1024,
                    )
                )
            except PaperSearchTimeoutError:
                raise
            except Exception as exc:
                attempt["crossref_failed_count"] += 1
                attempt.setdefault("crossref_transport_errors", []).append(
                    type(exc).__name__
                )
                continue
            status = int(response.get("status", 0))
            attempt["crossref_http_statuses"].append(status)
            if status != 200 or response.get("oversize"):
                attempt["crossref_failed_count"] += 1
                continue
            payload = response.get("json")
            if payload is None:
                try:
                    payload = json.loads(response.get("body", b"{}") or b"{}")
                except (TypeError, ValueError, UnicodeDecodeError):
                    payload = {}
            message = payload.get("message") if isinstance(payload, Mapping) else None
            items = message.get("items", []) if isinstance(message, Mapping) else []
            if not isinstance(items, list) or not items:
                attempt["crossref_failed_count"] += 1
                continue
            item = items[0]
            if not isinstance(item, Mapping):
                attempt["crossref_failed_count"] += 1
                continue
            item_title = item.get("title")
            if isinstance(item_title, list):
                item_title = item_title[0] if item_title else ""
            if not _crossref_title_matches(title, item_title):
                attempt["crossref_failed_count"] += 1
                continue
            doi = item.get("DOI") or item.get("doi")
            if isinstance(candidate, dict) and isinstance(doi, str) and doi.strip():
                candidate["doi"] = doi.strip().lower()
                if isinstance(item_title, str) and item_title.strip():
                    candidate["crossref_title"] = item_title.strip()
                containers = item.get("short-container-title")
                if isinstance(containers, list) and containers:
                    candidate["journal"] = str(containers[0])
                attempt["crossref_matched_count"] += 1

    @staticmethod
    def _reconstruct_openalex_abstract(candidate: Mapping[str, Any]) -> str:
        """从 OpenAlex 倒排摘要恢复用于本地排序的文本。"""

        inverted = candidate.get("abstract_inverted_index")
        if not isinstance(inverted, Mapping):
            return ""
        words: List[Tuple[int, str]] = []
        for token, positions in inverted.items():
            if not isinstance(token, str) or not isinstance(positions, list):
                continue
            for position in positions:
                if isinstance(position, int):
                    words.append((position, token))
        return " ".join(token for _, token in sorted(words))

    def _rank_openalex_candidates_locally(
        self,
        candidates: List[Mapping[str, Any]],
        ranking_question: Optional[str],
        *,
        audit_attempt: Optional[Dict[str, Any]] = None,
    ) -> List[Mapping[str, Any]]:
        """用已加载的本地 BGE 对候选标题和摘要排序，不设置科学阈值。"""

        if not ranking_question or not candidates:
            return candidates
        texts = []
        for candidate in candidates:
            title = candidate.get("display_name") or candidate.get("title") or ""
            abstract = (
                candidate.get("abstract")
                or candidate.get("snippet")
                or self._reconstruct_openalex_abstract(candidate)
                or ""
            )
            texts.append(f"{title}\n{abstract}".strip())
        try:
            vectors = self._get_embeddings([ranking_question, *texts])
        except PaperSearchTimeoutError:
            raise
        except Exception as exc:
            raise RetrievalEmbeddingError(
                f"OpenAlex 候选本地 BGE 排序失败：{type(exc).__name__}"
            ) from exc
        if vectors.ndim != 2 or vectors.shape[0] != len(candidates) + 1:
            raise RetrievalEmbeddingError("OpenAlex 候选本地 BGE 排序返回维度不一致")
        scores = np.matmul(vectors[1:], vectors[0])
        ranked = [
            (float(score), candidate)
            for score, candidate in zip(scores, candidates)
        ]
        ranked.sort(key=lambda item: item[0], reverse=True)
        if audit_attempt is not None:
            audit_attempt["ranking_scores"] = [
                {
                    "identifiers": list(_normalize_candidate_identifier(candidate)),
                    "score": round(score, 8),
                }
                for score, candidate in ranked
            ]
        for score, candidate in ranked:
            if isinstance(candidate, dict):
                candidate["_arche_local_similarity"] = score
        return [candidate for _, candidate in ranked]

    def _request_openalex_candidates(
        self,
        keyword: str,
        *,
        openalex_per_page: int,
        timeout_seconds: float,
        attempt: Dict[str, Any],
        openalex_state: Dict[str, Any],
    ) -> List[Mapping[str, Any]]:
        """用一次最小字段 OpenAlex 查询取得 OA 且非撤稿候选。"""

        remaining = openalex_state.get("remaining")
        if remaining is not None and remaining < 10:
            attempt["skipped_reason"] = "provider_budget_insufficient"
            return []
        params: Dict[str, Any] = {
            "search": keyword,
            "filter": "open_access.is_oa:true,has_pdf_url:true,is_retracted:false",
            "per_page": min(100, openalex_per_page),
            "select": (
                "id,doi,display_name,type,is_retracted,open_access,"
                "best_oa_location,primary_location,locations,abstract_inverted_index"
            ),
        }
        api_key = os.environ.get("OPENALEX_API_KEY", "").strip()
        headers = {"User-Agent": "ARCHE-Memory/1 paper-search"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        # API key 可进行有限重试；匿名共享池遇到 429 立即停止，5xx 只允许一次
        # 很短的恢复尝试。每次响应只保留状态和数字预算，绝不记录请求头。
        max_retries = 2 if api_key else 1
        retry_delay = 1.0
        for retry_index in range(max_retries + 1):
            if retry_index > 0:
                self._sleep_paper_search_with_backoff(retry_delay)
            try:
                response = _normalize_paper_search_http_response(
                    _paper_search_http_request(
                        OPENALEX_SEARCH_URL,
                        params=params,
                        headers=headers,
                        timeout_seconds=min(timeout_seconds, 30.0),
                        max_bytes=8 * 1024 * 1024,
                    )
                )
            except PaperSearchTimeoutError:
                raise
            except Exception as exc:
                attempt["transport_error"] = type(exc).__name__
                return []
            status = int(response.get("status", 0))
            attempt["http_statuses"].append(status)
            if response.get("oversize"):
                attempt["response_failure"] = "oversize"
                return []
            budget = _safe_http_budget(response.get("headers", {}))
            if any(value is not None for value in budget.values()):
                attempt["budget"] = budget
                if budget.get("remaining") is not None:
                    openalex_state["remaining"] = budget["remaining"]
            if status == 200:
                payload = response.get("json")
                if payload is None:
                    try:
                        payload = json.loads(response.get("body", b"{}") or b"{}")
                    except (TypeError, ValueError, UnicodeDecodeError):
                        payload = {}
                candidates = payload.get("results", []) if isinstance(payload, Mapping) else []
                if not isinstance(candidates, list):
                    candidates = []
                normalized = []
                for item in candidates:
                    if not isinstance(item, Mapping):
                        continue
                    if item.get("is_retracted") is True:
                        continue
                    open_access = item.get("open_access")
                    if isinstance(open_access, Mapping) and open_access.get("is_oa") is False:
                        continue
                    if not _candidate_pdf_urls(item):
                        continue
                    normalized.append(item)
                attempt["candidate_count"] = len(normalized)
                self._paper_search_audit["candidate_count"] += len(normalized)
                return normalized
            retryable = status == 429 or 500 <= status < 600
            if status == 429 and not api_key:
                attempt["shared_pool_limited"] = True
                retry_after = _parse_retry_after_seconds(response.get("headers", {}))
                if retry_after is not None:
                    attempt["retry_after_seconds"].append(retry_after)
                break
            if not retryable or retry_index >= max_retries:
                break
            retry_after = _parse_retry_after_seconds(response.get("headers", {}))
            if retry_after is not None:
                attempt["retry_after_seconds"].append(retry_after)
            retry_delay = max(retry_delay * 2.0, retry_after or 0.0, 1.0)
            attempt["retry_count"] += 1
        return []

    def _download_paper_candidate_pdf(
        self,
        candidate: Mapping[str, Any],
        pdf_dir: str,
        *,
        known_hashes: Dict[str, str],
        seen_identifiers: set[str],
        attempt: Dict[str, Any],
        timeout_seconds: float,
        download_delay_seconds: float,
        max_pdf_bytes: int,
        download_state: Dict[str, float],
        keyword: str,
    ) -> Optional[str]:
        """按候选 OA URL 顺序下载，校验后原子写入并按 ID/SHA 去重。"""

        identifiers = _normalize_candidate_identifier(candidate)
        if identifiers and any(identifier in seen_identifiers for identifier in identifiers):
            attempt["deduplicated_count"] += 1
            self._paper_search_audit["deduplicated_count"] += 1
            return None
        urls = _candidate_pdf_urls(candidate)
        if not urls:
            attempt["failed_download_count"] += 1
            self._paper_search_audit["failed_download_count"] += 1
            return None
        for url in urls:
            last_download_at = download_state.get("last_request_at")
            if last_download_at is not None:
                elapsed = self._paper_search_clock() - float(last_download_at)
                if elapsed < download_delay_seconds:
                    self._sleep_paper_search_with_backoff(download_delay_seconds - elapsed)
            download_state["last_request_at"] = self._paper_search_clock()
            try:
                response = _normalize_paper_search_http_response(
                    _paper_search_http_request(
                        url,
                        headers={"User-Agent": "ARCHE-Memory/1 paper-download"},
                        timeout_seconds=min(timeout_seconds, 30.0),
                        max_bytes=max_pdf_bytes,
                    )
                )
            except PaperSearchTimeoutError:
                raise
            except Exception as exc:
                self._paper_search_audit.setdefault("download_attempts", []).append(
                    {"provider": attempt["provider"], "status": "transport_error", "error": type(exc).__name__}
                )
                continue
            status = int(response.get("status", 0))
            self._paper_search_audit.setdefault("download_attempts", []).append(
                {"provider": attempt["provider"], "status": status}
            )
            if status != 200:
                continue
            if response.get("oversize"):
                self._paper_search_audit.setdefault("download_attempts", [])[-1][
                    "failure"
                ] = "oversize"
                continue
            body = response.get("body", b"")
            if not isinstance(body, (bytes, bytearray)):
                body = str(body).encode("utf-8", errors="ignore")
            body = bytes(body)
            if len(body) > max_pdf_bytes:
                self._paper_search_audit.setdefault("download_attempts", [])[-1]["failure"] = "oversize"
                continue
            if not _validate_downloaded_pdf_bytes(body):
                self._paper_search_audit.setdefault("download_attempts", [])[-1]["failure"] = "invalid_pdf"
                continue
            digest = hashlib.sha256(body).hexdigest()
            if digest in known_hashes:
                attempt["deduplicated_count"] += 1
                self._paper_search_audit["deduplicated_count"] += 1
                for identifier in identifiers:
                    seen_identifiers.add(identifier)
                return None
            filename = f"{digest}.pdf"
            destination = os.path.abspath(os.path.join(pdf_dir, filename))
            os.makedirs(pdf_dir, exist_ok=True)
            fd, temporary_path = tempfile.mkstemp(
                prefix=f".{digest}.", suffix=".part", dir=pdf_dir
            )
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(body)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_path, destination)
            finally:
                if os.path.exists(temporary_path):
                    os.unlink(temporary_path)
            known_hashes[digest] = destination
            for identifier in identifiers:
                seen_identifiers.add(identifier)
            attempt["download_count"] += 1
            self._paper_search_audit["download_count"] += 1
            accepted_record: Dict[str, Any] = {
                "path": destination,
                "sha256": digest,
                "provider": attempt["provider"],
                "keyword": keyword,
                "identifiers": list(identifiers),
                "source_host": urlsplit(url).hostname or "",
            }
            title = candidate.get("display_name") or candidate.get("title")
            if isinstance(title, str) and title.strip():
                accepted_record["title"] = title.strip()
            similarity = candidate.get("_arche_local_similarity")
            if isinstance(similarity, (int, float)):
                accepted_record["local_similarity"] = round(float(similarity), 8)
            self._paper_search_audit.setdefault("accepted_papers", []).append(
                accepted_record
            )
            return destination
        attempt["failed_download_count"] += 1
        self._paper_search_audit["failed_download_count"] += 1
        return None

    def _search_single_keyword_with_provider_strategy(
        self,
        *,
        keyword: str,
        pdf_dir: str,
        target_papers: int,
        keyword_target_papers: Optional[int] = None,
        known_hashes: Dict[str, str],
        seen_identifiers: set[str],
        download_state: Dict[str, float],
        scholar_state: Dict[str, Any],
        crossref_state: Dict[str, Any],
        semantic_scholar_state: Dict[str, Any],
        openalex_state: Dict[str, Any],
        has_s2_key: bool,
        scholar_allowed: bool,
        provider_attempts_allowed: bool,
        openalex_allowed: bool,
        candidate_page_size: int,
        scholar_max_pages: int,
        scholar_results_per_page: int,
        scholar_crossref_max_candidates: int,
        completed_scholar_pages: Optional[Dict[str, set[int]]],
        attempted_scholar_pages: Optional[Dict[str, set[int]]],
        scholar_progress_sink: Optional[Callable[[str, int, bool], None]],
        timeout: float,
        ranking_question: Optional[str],
        openalex_per_page: int,
        download_delay_seconds: float,
        max_pdf_bytes: int,
    ) -> Dict[str, Any]:
        """执行一个关键词的 provider 查询和 PDF 下载，不处理跨关键词状态。"""

        keyword_target = (
            target_papers
            if keyword_target_papers is None
            else min(target_papers, int(keyword_target_papers))
        )
        keyword_status: Dict[str, Any] = {
            "keyword": keyword,
            "status": "failed",
            "downloaded_files": [],
            "provider": None,
        }
        if scholar_allowed:
            for scholar_page in range(scholar_max_pages):
                if scholar_page in (completed_scholar_pages or {}).get(keyword, set()):
                    continue
                # 记录 attempted 与 completed 分开：只有收到明确响应并完成解析后才
                # 将页码加入 completed；请求中断时恢复会重新尝试该页而不会永久漏页。
                if attempted_scholar_pages is not None:
                    attempted_scholar_pages.setdefault(keyword, set()).add(scholar_page)
                if scholar_progress_sink is not None:
                    scholar_progress_sink(keyword, scholar_page, False)
                scholar_attempt = self._create_paper_search_provider_attempt(
                    "pypaperbot_scholar", keyword
                )
                candidates = self._request_pypaperbot_scholar_candidates(
                    keyword,
                    scholar_page=scholar_page,
                    scholar_results_per_page=scholar_results_per_page,
                    scholar_crossref_max_candidates=scholar_crossref_max_candidates,
                    timeout_seconds=timeout,
                    attempt=scholar_attempt,
                    scholar_state=scholar_state,
                    crossref_state=crossref_state,
                )
                if not candidates:
                    if scholar_progress_sink is not None:
                        scholar_progress_sink(keyword, scholar_page, True)
                    break
                keyword_status["provider"] = "pypaperbot_scholar"
                candidates = self._rank_openalex_candidates_locally(
                    list(candidates),
                    ranking_question,
                    audit_attempt=scholar_attempt,
                )
                for candidate in candidates:
                    if len(known_hashes) >= keyword_target:
                        keyword_status["stop_reason"] = "keyword_allocation_reached"
                        break
                    path = self._download_paper_candidate_pdf(
                        candidate,
                        pdf_dir,
                        known_hashes=known_hashes,
                        seen_identifiers=seen_identifiers,
                        attempt=scholar_attempt,
                        timeout_seconds=timeout,
                        download_delay_seconds=download_delay_seconds,
                        max_pdf_bytes=max_pdf_bytes,
                        download_state=download_state,
                        keyword=keyword,
                    )
                    if path:
                        keyword_status["downloaded_files"].append(path)
                if len(known_hashes) >= keyword_target:
                    if len(known_hashes) >= target_papers:
                        self._paper_search_audit["target_stop_reached"] = True
                    else:
                        keyword_status["stop_reason"] = "keyword_allocation_reached"
                    if scholar_progress_sink is not None:
                        scholar_progress_sink(keyword, scholar_page, True)
                    break
                if len(candidates) < scholar_results_per_page:
                    if scholar_progress_sink is not None:
                        scholar_progress_sink(keyword, scholar_page, True)
                    break
                if scholar_progress_sink is not None:
                    scholar_progress_sink(keyword, scholar_page, False)
        candidates: List[Mapping[str, Any]] = []
        if provider_attempts_allowed:
            s2_attempt = self._create_paper_search_provider_attempt(
                "semantic_scholar", keyword
            )
            candidates = self._request_semantic_scholar_candidates(
                keyword,
                candidate_page_size=candidate_page_size,
                timeout_seconds=timeout,
                attempt=s2_attempt,
                has_api_key=has_s2_key,
                semantic_scholar_state=semantic_scholar_state,
            )
            if candidates:
                keyword_status["provider"] = "semantic_scholar"
                candidates = self._rank_openalex_candidates_locally(
                    candidates, ranking_question
                )
                for candidate in candidates:
                    if len(known_hashes) >= keyword_target:
                        break
                    path = self._download_paper_candidate_pdf(
                        candidate,
                        pdf_dir,
                        known_hashes=known_hashes,
                        seen_identifiers=seen_identifiers,
                        attempt=s2_attempt,
                        timeout_seconds=timeout,
                        download_delay_seconds=download_delay_seconds,
                        max_pdf_bytes=max_pdf_bytes,
                        download_state=download_state,
                        keyword=keyword,
                    )
                    if path:
                        keyword_status["downloaded_files"].append(path)
        if openalex_allowed and len(known_hashes) < keyword_target:
            oa_attempt = self._create_paper_search_provider_attempt("openalex", keyword)
            oa_candidates = self._request_openalex_candidates(
                keyword,
                openalex_per_page=openalex_per_page,
                timeout_seconds=timeout,
                attempt=oa_attempt,
                openalex_state=openalex_state,
            )
            oa_candidates = self._rank_openalex_candidates_locally(
                oa_candidates,
                ranking_question,
                audit_attempt=oa_attempt,
            )
            if oa_candidates:
                keyword_status["provider"] = "openalex"
            for candidate in oa_candidates:
                if len(known_hashes) >= keyword_target:
                    if len(known_hashes) >= target_papers:
                        self._paper_search_audit["target_stop_reached"] = True
                    else:
                        keyword_status["stop_reason"] = "keyword_allocation_reached"
                    break
                path = self._download_paper_candidate_pdf(
                    candidate,
                    pdf_dir,
                    known_hashes=known_hashes,
                    seen_identifiers=seen_identifiers,
                    attempt=oa_attempt,
                    timeout_seconds=timeout,
                    download_delay_seconds=download_delay_seconds,
                    max_pdf_bytes=max_pdf_bytes,
                    download_state=download_state,
                    keyword=keyword,
                )
                if path:
                    keyword_status["downloaded_files"].append(path)
        if keyword_status["downloaded_files"]:
            keyword_status["status"] = "success"
        elif keyword_status["provider"]:
            keyword_status["status"] = "no_valid_pdf"
        else:
            keyword_status["status"] = "failed"
        keyword_status["target_count"] = len(known_hashes)
        return keyword_status

    def _search_papers_with_provider_strategy(
        self,
        *,
        keywords: List[str],
        pdf_dir: str,
        limit_per_keyword: int,
        status_sink: Optional[Callable[[Mapping[str, Any]], None]],
        timeout_seconds: Optional[float],
        target_papers: Optional[int],
        keyword_delay_seconds: Optional[float],
        candidate_page_size: int,
        provider_strategy: str,
        ranking_question: Optional[str],
        openalex_per_page: int,
        download_delay_seconds: Optional[float],
        scholar_max_pages: int,
        scholar_results_per_page: int,
        scholar_crossref_max_candidates: int,
        scholar_request_delay_seconds: Optional[float],
        scholar_request_jitter_seconds: Optional[float],
        scholar_initial_delay_seconds: Optional[float],
        scholar_initial_jitter_seconds: Optional[float],
        scholar_backoff_base_seconds: Optional[float],
        scholar_max_backoff_seconds: Optional[float],
        scholar_max_consecutive_rate_limits: Optional[int],
        keyword_jitter_seconds: Optional[float],
        resume_state_path: Optional[str],
        response_cache_dir: Optional[str],
    ) -> List[str]:
        """运行 Scholar 主路径或兼容 provider，并在 PaperSearch 边界返回 PDF。"""

        keywords = list(dict.fromkeys(keywords))
        if not keywords:
            raise ValueError("论文检索至少需要一个关键词")
        if target_papers is None:
            target_papers = max(1, limit_per_keyword * len(keywords))
        if target_papers <= 0:
            raise ValueError("全局论文目标必须为正整数")
        if not isinstance(openalex_per_page, int) or isinstance(openalex_per_page, bool) or not 1 <= openalex_per_page <= 100:
            raise ValueError("OpenAlex 每页候选数量必须在1到100之间")
        if (
            not isinstance(scholar_max_pages, int)
            or isinstance(scholar_max_pages, bool)
            or not 1 <= scholar_max_pages <= 20
        ):
            raise ValueError("Google Scholar 最大页数必须在1到20之间")
        if (
            not isinstance(scholar_results_per_page, int)
            or isinstance(scholar_results_per_page, bool)
            or not 1 <= scholar_results_per_page <= 20
        ):
            raise ValueError("Google Scholar 每页结果数必须在1到20之间")
        if (
            not isinstance(scholar_crossref_max_candidates, int)
            or isinstance(scholar_crossref_max_candidates, bool)
            or not 0 <= scholar_crossref_max_candidates <= 20
        ):
            raise ValueError("Scholar 每页 Crossref 补充上限必须在0到20之间")
        timeout = float(timeout_seconds if timeout_seconds is not None else 180.0)
        if timeout <= 0:
            raise ValueError("单关键词 PaperSearch 墙钟上限必须为正数")
        if download_delay_seconds is None:
            try:
                download_delay_seconds = float(os.environ.get("ARCHE_PAPERSEARCH_DOWNLOAD_DELAY_SECONDS", "1"))
            except ValueError:
                download_delay_seconds = 1.0
        if keyword_delay_seconds is None:
            try:
                keyword_delay_seconds = float(os.environ.get("ARCHE_PAPERSEARCH_KEYWORD_DELAY_SECONDS", "45"))
            except ValueError:
                keyword_delay_seconds = 45.0
        if scholar_request_delay_seconds is None:
            try:
                scholar_request_delay_seconds = float(
                    os.environ.get("ARCHE_PAPERSEARCH_SCHOLAR_REQUEST_DELAY_SECONDS", "15")
                )
            except ValueError:
                scholar_request_delay_seconds = 15.0
        if scholar_request_jitter_seconds is None:
            try:
                scholar_request_jitter_seconds = float(
                    os.environ.get("ARCHE_PAPERSEARCH_SCHOLAR_REQUEST_JITTER_SECONDS", "15")
                )
            except ValueError:
                scholar_request_jitter_seconds = 15.0
        if scholar_initial_delay_seconds is None:
            try:
                scholar_initial_delay_seconds = float(
                    os.environ.get("ARCHE_PAPERSEARCH_SCHOLAR_INITIAL_DELAY_SECONDS", "10")
                )
            except ValueError:
                scholar_initial_delay_seconds = 10.0
        if scholar_initial_jitter_seconds is None:
            try:
                scholar_initial_jitter_seconds = float(
                    os.environ.get(
                        "ARCHE_PAPERSEARCH_SCHOLAR_INITIAL_JITTER_SECONDS", "20"
                    )
                )
            except ValueError:
                scholar_initial_jitter_seconds = 20.0
        if scholar_backoff_base_seconds is None:
            try:
                scholar_backoff_base_seconds = float(
                    os.environ.get("ARCHE_PAPERSEARCH_SCHOLAR_BACKOFF_BASE_SECONDS", "900")
                )
            except ValueError:
                scholar_backoff_base_seconds = 900.0
        if scholar_max_backoff_seconds is None:
            try:
                scholar_max_backoff_seconds = float(
                    os.environ.get("ARCHE_PAPERSEARCH_SCHOLAR_MAX_BACKOFF_SECONDS", "3600")
                )
            except ValueError:
                scholar_max_backoff_seconds = 3600.0
        if scholar_max_consecutive_rate_limits is None:
            try:
                scholar_max_consecutive_rate_limits = int(
                    os.environ.get(
                        "ARCHE_PAPERSEARCH_SCHOLAR_MAX_CONSECUTIVE_RATE_LIMITS", "1"
                    )
                )
            except ValueError:
                scholar_max_consecutive_rate_limits = 1
        if keyword_jitter_seconds is None:
            try:
                default_keyword_jitter = "45" if keyword_delay_seconds > 0 else "0"
                keyword_jitter_seconds = float(
                    os.environ.get(
                        "ARCHE_PAPERSEARCH_KEYWORD_JITTER_SECONDS",
                        default_keyword_jitter,
                    )
                )
            except ValueError:
                keyword_jitter_seconds = 0.0
        if download_delay_seconds < 0:
            raise ValueError("PDF 下载间隔不能为负数")
        if keyword_delay_seconds < 0:
            raise ValueError("关键词退避秒数不能为负数")
        for config_name, config_value in (
            ("Scholar 页面间隔", scholar_request_delay_seconds),
            ("Scholar 页面随机抖动", scholar_request_jitter_seconds),
            ("Scholar 首次预等待", scholar_initial_delay_seconds),
            ("Scholar 首次预等待随机抖动", scholar_initial_jitter_seconds),
            ("Scholar 限流基础退避", scholar_backoff_base_seconds),
            ("Scholar 最大冷却", scholar_max_backoff_seconds),
            ("关键词随机抖动", keyword_jitter_seconds),
        ):
            if float(config_value) < 0:
                raise ValueError(f"{config_name}不能为负数")
        if (
            not isinstance(scholar_max_consecutive_rate_limits, int)
            or isinstance(scholar_max_consecutive_rate_limits, bool)
            or scholar_max_consecutive_rate_limits < 1
        ):
            raise ValueError("Scholar 连续限流停机阈值必须为正整数")
        if provider_strategy == "pypaperbot_scholar":
            if resume_state_path is None:
                resume_state_path = os.path.join(pdf_dir, ".paper-search-resume.json")
            if response_cache_dir is None:
                response_cache_dir = os.path.join(pdf_dir, ".paper-search-cache")
        effective_search_config = {
            "scholar_request_delay_seconds": float(scholar_request_delay_seconds),
            "scholar_request_jitter_seconds": float(scholar_request_jitter_seconds),
            "scholar_initial_delay_seconds": float(scholar_initial_delay_seconds),
            "scholar_initial_jitter_seconds": float(scholar_initial_jitter_seconds),
            "scholar_backoff_base_seconds": float(scholar_backoff_base_seconds),
            "scholar_max_backoff_seconds": float(scholar_max_backoff_seconds),
            "scholar_max_consecutive_rate_limits": int(scholar_max_consecutive_rate_limits),
            "keyword_delay_seconds": float(keyword_delay_seconds),
            "keyword_jitter_seconds": float(keyword_jitter_seconds),
        }
        try:
            max_pdf_bytes = int(
                os.environ.get(
                    "ARCHE_PAPERSEARCH_MAX_PDF_BYTES", str(50 * 1024 * 1024)
                )
            )
        except ValueError as exc:
            raise ValueError("PDF 最大字节数必须为正整数") from exc
        if max_pdf_bytes <= 0:
            raise ValueError("PDF 最大字节数必须为正整数")
        os.makedirs(pdf_dir, exist_ok=True)
        self._paper_search_audit = {
            "provider_attempts": [],
            "download_attempts": [],
            "candidate_count": 0,
            "download_count": 0,
            "deduplicated_count": 0,
            "failed_download_count": 0,
            "target_stop_reached": False,
            "max_pdf_bytes": max_pdf_bytes,
            "accepted_papers": [],
            "effective_pacing": effective_search_config,
            "keyword_allocations": {},
            "keyword_allocation_history": [],
            "keyword_allocations_applied": True,
            "global_target_cap": target_papers,
        }
        stale_parts = 0
        for filename in sorted(os.listdir(pdf_dir)):
            if not filename.endswith(".part"):
                continue
            stale_path = os.path.join(pdf_dir, filename)
            if not os.path.isfile(stale_path):
                continue
            try:
                os.unlink(stale_path)
                stale_parts += 1
            except OSError:
                logger.warning("无法清理残留 PaperSearch 临时文件：%s", stale_path)
        self._paper_search_audit["stale_part_files_removed"] = stale_parts
        existing_paths = self._collect_valid_pdf_paths(pdf_dir)
        known_hashes: Dict[str, str] = {}
        for path in existing_paths:
            try:
                with open(path, "rb") as handle:
                    digest = hashlib.sha256(handle.read()).hexdigest()
                    if digest not in known_hashes:
                        known_hashes[digest] = path
                        self._paper_search_audit["accepted_papers"].append(
                            {
                                "path": path,
                                "sha256": digest,
                                "provider": "resume_existing",
                                "keyword": None,
                                "identifiers": [],
                                "source_host": "",
                            }
                        )
            except OSError:
                continue
        current_corpus_records = self._collect_pdf_corpus_records(pdf_dir)
        current_corpus_fingerprint = self._calculate_pdf_corpus_fingerprint(pdf_dir)
        resume_state = self._load_paper_search_resume_state(resume_state_path)
        resume_matches = (
            resume_state.get("provider_strategy") == provider_strategy
            and resume_state.get("keywords") == keywords
            and resume_state.get("target_papers") == target_papers
            and resume_state.get("corpus_fingerprint") == current_corpus_fingerprint
            and resume_state.get("corpus_files") == current_corpus_records
        )
        completed_scholar_pages: Dict[str, set[int]] = {}
        completed_keywords: set[str] = set()
        next_keyword_index = 0
        if resume_matches:
            raw_pages = resume_state.get("completed_pages", {})
            if isinstance(raw_pages, Mapping):
                for raw_keyword, raw_values in raw_pages.items():
                    if raw_keyword not in keywords or not isinstance(raw_values, list):
                        continue
                    completed_scholar_pages[raw_keyword] = {
                        int(page) for page in raw_values if isinstance(page, int)
                    }
            attempted_scholar_pages: Dict[str, set[int]] = {}
            raw_attempted_pages = resume_state.get("attempted_pages", {})
            if isinstance(raw_attempted_pages, Mapping):
                for raw_keyword, raw_values in raw_attempted_pages.items():
                    if raw_keyword not in keywords or not isinstance(raw_values, list):
                        continue
                    attempted_scholar_pages[raw_keyword] = {
                        int(page) for page in raw_values if isinstance(page, int)
                    }
            completed_keywords = {
                value
                for value in resume_state.get("completed_keywords", [])
                if isinstance(value, str) and value in keywords
            }
            try:
                next_keyword_index = max(
                    0,
                    min(len(keywords), int(resume_state.get("next_keyword_index", 0))),
                )
            except (TypeError, ValueError):
                next_keyword_index = 0
            if isinstance(resume_state.get("scholar_state"), Mapping):
                resume_scholar_state = dict(resume_state["scholar_state"])
            else:
                resume_scholar_state = {}
        else:
            resume_scholar_state = {}
            attempted_scholar_pages = {}
        seen_identifiers: set[str] = set()
        if resume_matches:
            for accepted in resume_state.get("accepted_papers", []):
                if not isinstance(accepted, Mapping):
                    continue
                identifiers = accepted.get("identifiers", [])
                if isinstance(identifiers, list):
                    seen_identifiers.update(
                        value.strip()
                        for value in identifiers
                        if isinstance(value, str) and value.strip()
                    )
        download_state: Dict[str, float] = {}
        scholar_state: Dict[str, Any] = {
            "blocked": False,
            "request_delay_seconds": float(scholar_request_delay_seconds),
            "request_jitter_seconds": float(scholar_request_jitter_seconds),
            "initial_delay_seconds": float(scholar_initial_delay_seconds),
            "initial_jitter_seconds": float(scholar_initial_jitter_seconds),
            "backoff_base_seconds": float(scholar_backoff_base_seconds),
            "max_backoff_seconds": float(scholar_max_backoff_seconds),
            "max_consecutive_rate_limits": int(scholar_max_consecutive_rate_limits),
            "consecutive_rate_limits": 0,
            "response_cache_dir": response_cache_dir,
        }
        scholar_state.update(
            {
                key: value
                for key, value in resume_scholar_state.items()
                if key in {
                    "blocked",
                    "blocked_reason",
                    "blocked_failure",
                    "cooldown_until",
                    "consecutive_rate_limits",
                }
            }
        )
        cooldown_until = scholar_state.get("cooldown_until")
        if (
            scholar_state.get("blocked_failure") == "rate_limited_shutdown"
            and isinstance(cooldown_until, (int, float))
        ):
            cooldown_remaining = float(cooldown_until) - time.time()
            if cooldown_remaining > 0:
                scholar_state["cooldown_remaining"] = cooldown_remaining
                scholar_state["blocked"] = True
            else:
                scholar_state["blocked"] = False
                scholar_state.pop("blocked_reason", None)
                scholar_state.pop("blocked_failure", None)
                scholar_state.pop("cooldown_until", None)
        crossref_state: Dict[str, Any] = {"request_interval_seconds": 1.0}
        semantic_scholar_state: Dict[str, Any] = {"anonymous_disabled": False, "next_delay": 1.0}
        openalex_state: Dict[str, Any] = {"remaining": None}
        has_s2_key = bool(os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip())
        scholar_allowed = provider_strategy == "pypaperbot_scholar"
        provider_attempts_allowed = provider_strategy in {"semantic_scholar_then_openalex", "semantic_scholar"}
        openalex_allowed = provider_strategy in {"semantic_scholar_then_openalex", "openalex"}

        def save_resume_state(keyword_index: int) -> None:
            self._write_paper_search_resume_state(
                resume_state_path,
                self._build_paper_search_resume_snapshot(
                    keywords=keywords,
                    provider_strategy=provider_strategy,
                    target_papers=target_papers,
                    corpus_files=self._collect_pdf_corpus_records(pdf_dir),
                    corpus_fingerprint=self._calculate_pdf_corpus_fingerprint(pdf_dir),
                    next_keyword_index=keyword_index,
                    completed_pages=completed_scholar_pages,
                    attempted_pages=attempted_scholar_pages,
                    completed_keywords=completed_keywords,
                    accepted_papers=self._paper_search_audit.get("accepted_papers", []),
                    scholar_state=scholar_state,
                    response_cache_dir=response_cache_dir,
                ),
            )

        def record_scholar_page(keyword: str, page: int, _finished: bool) -> None:
            if _finished:
                completed_scholar_pages.setdefault(keyword, set()).add(int(page))
            save_resume_state(next_keyword_index)

        if response_cache_dir:
            os.makedirs(response_cache_dir, exist_ok=True)

        for keyword_index, keyword in enumerate(keywords):
            next_keyword_index = keyword_index
            if keyword in completed_keywords:
                continue
            if len(known_hashes) >= target_papers:
                self._paper_search_audit["target_stop_reached"] = True
                break
            remaining_keywords = [
                item for item in keywords if item not in completed_keywords
            ]
            remaining_gap = max(target_papers - len(known_hashes), 0)
            allocation = allocate_fair_paper_search_limits(
                remaining_keywords,
                remaining_gap,
            )[keyword]
            keyword_target_papers = min(
                target_papers,
                len(known_hashes) + allocation,
            )
            self._paper_search_audit["keyword_allocations"][keyword] = allocation
            self._paper_search_audit["keyword_allocation_history"].append(
                {
                    "keyword": keyword,
                    "remaining_before": remaining_gap,
                    "remaining_keywords": len(remaining_keywords),
                    "allocated_limit": allocation,
                }
            )
            if allocation <= 0:
                keyword_status = {
                    "keyword": keyword,
                    "status": "skipped_no_allocation",
                    "downloaded_files": [],
                    "provider": None,
                    "allocated_limit": 0,
                    "actual_added": 0,
                    "remaining_after": remaining_gap,
                    "target_count": len(known_hashes),
                }
                if status_sink is not None:
                    status_sink(dict(keyword_status))
                completed_keywords.add(keyword)
                next_keyword_index = keyword_index + 1
                save_resume_state(next_keyword_index)
                continue
            hashes_before_keyword = set(known_hashes)
            try:
                with _limit_single_keyword_paper_search_time(timeout):
                    keyword_status = self._search_single_keyword_with_provider_strategy(
                        keyword=keyword,
                        pdf_dir=pdf_dir,
                        target_papers=target_papers,
                        keyword_target_papers=keyword_target_papers,
                        known_hashes=known_hashes,
                        seen_identifiers=seen_identifiers,
                        download_state=download_state,
                        scholar_state=scholar_state,
                        crossref_state=crossref_state,
                        semantic_scholar_state=semantic_scholar_state,
                        openalex_state=openalex_state,
                        has_s2_key=has_s2_key,
                        scholar_allowed=scholar_allowed,
                        provider_attempts_allowed=provider_attempts_allowed,
                        openalex_allowed=openalex_allowed,
                        candidate_page_size=candidate_page_size,
                        scholar_max_pages=scholar_max_pages,
                        scholar_results_per_page=scholar_results_per_page,
                        scholar_crossref_max_candidates=scholar_crossref_max_candidates,
                        completed_scholar_pages=completed_scholar_pages,
                        attempted_scholar_pages=attempted_scholar_pages,
                        scholar_progress_sink=record_scholar_page,
                        timeout=timeout,
                        ranking_question=ranking_question,
                        openalex_per_page=openalex_per_page,
                        download_delay_seconds=float(download_delay_seconds),
                        max_pdf_bytes=max_pdf_bytes,
                    )
            except PaperSearchTimeoutError as exc:
                partial_paths = sorted(
                    path
                    for digest, path in known_hashes.items()
                    if digest not in hashes_before_keyword
                )
                keyword_status = {
                    "keyword": keyword,
                    "status": "timeout",
                    "downloaded_files": partial_paths,
                    "provider": None,
                    "timeout_seconds": timeout,
                    "error": str(exc),
                    "target_count": len(known_hashes),
                }
            keyword_status["allocated_limit"] = allocation
            keyword_status["actual_added"] = max(
                len(known_hashes) - len(hashes_before_keyword), 0
            )
            keyword_status["remaining_after"] = max(
                target_papers - len(known_hashes), 0
            )
            if status_sink is not None:
                status_sink(dict(keyword_status))
            completed_keywords.add(keyword)
            next_keyword_index = keyword_index + 1
            save_resume_state(next_keyword_index)
            if len(known_hashes) >= target_papers:
                self._paper_search_audit["target_stop_reached"] = True
                break
            if scholar_allowed and scholar_state.get("blocked"):
                break
            if keyword_delay_seconds and keyword_delay_seconds > 0:
                self._sleep_paper_search_with_backoff(
                    float(keyword_delay_seconds)
                    + self._sample_paper_search_jitter(float(keyword_jitter_seconds))
                )
        save_resume_state(len(keywords))
        return sorted(known_hashes.values())

    def search_papers(
        self,
        keywords: List[str],
        pdf_dir: str,
        limit_per_keyword: int = 3,
        status_sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
        timeout_seconds: Optional[float] = None,
        target_papers: Optional[int] = None,
        keyword_delay_seconds: Optional[float] = None,
        candidate_page_size: int = 100,
        provider_strategy: str = "paper_scraper",
        ranking_question: Optional[str] = None,
        openalex_per_page: int = 100,
        download_delay_seconds: Optional[float] = None,
        scholar_max_pages: int = 3,
        scholar_results_per_page: int = 10,
        scholar_crossref_max_candidates: int = 3,
        scholar_request_delay_seconds: Optional[float] = None,
        scholar_request_jitter_seconds: Optional[float] = None,
        scholar_initial_delay_seconds: Optional[float] = None,
        scholar_initial_jitter_seconds: Optional[float] = None,
        scholar_backoff_base_seconds: Optional[float] = None,
        scholar_max_backoff_seconds: Optional[float] = None,
        scholar_max_consecutive_rate_limits: Optional[int] = None,
        keyword_jitter_seconds: Optional[float] = None,
        resume_state_path: Optional[str] = None,
        response_cache_dir: Optional[str] = None,
    ) -> List[str]:
        """
        检索并下载相关论文

        参数:
            keywords: 关键词列表
            pdf_dir: PDF存储目录
            limit_per_keyword: 旧 paper-scraper 路径中每个关键词请求的论文目标
            status_sink: 可选的逐关键词结果记录器
            timeout_seconds: 单个关键词的墙钟上限，默认读取 ARCHE_PAPERSEARCH_TIMEOUT_SECONDS
            target_papers: 全局去重 PDF 目标；新 provider 路径达到目标后立即停止
            keyword_delay_seconds: 关键词间退避秒数，默认读取 ARCHE_PAPERSEARCH_KEYWORD_DELAY_SECONDS
            candidate_page_size: Semantic Scholar 每页候选数量；旧路径传给 ``_limit``
            provider_strategy: ``pypaperbot_scholar`` 使用 Google Scholar 元数据和页面
                明确暴露的公开 PDF；其它值是显式兼容路径。
            ranking_question: 可选原始问题，用本地 BGE 对 OpenAlex 标题/摘要排序。
            openalex_per_page: OpenAlex 单次元数据请求大小，最大 100。
            download_delay_seconds: OA 下载之间的最小间隔，默认读取环境变量。
            scholar_max_pages: 每个关键词最多读取的 Scholar 页数（1–20）。
            scholar_results_per_page: Scholar 每页结果数（1–20）。
            scholar_crossref_max_candidates: 每个 Scholar 页最多查询的 Crossref 元数据候选数（0–20）。
            scholar_request_delay_seconds: Scholar 页面之间的最小等待秒数。
            scholar_request_jitter_seconds: 每次 Scholar 页面请求前的随机抖动上限。
            scholar_initial_delay_seconds: 首次 Scholar 请求前的预等待秒数。
            scholar_initial_jitter_seconds: 首次 Scholar 请求前的随机抖动上限。
            scholar_backoff_base_seconds: 403/429 指数退避的基础秒数。
            scholar_max_backoff_seconds: 单次限流冷却的最大秒数。
            scholar_max_consecutive_rate_limits: 连续 403/429 后停机的次数。
            keyword_jitter_seconds: 关键词之间随机抖动上限。
            resume_state_path: 关键词/页码/已接受文献恢复文件，默认位于 PDF 目录。
            response_cache_dir: Scholar 200 HTML 响应缓存目录，默认位于 PDF 目录。

        返回:
            下载的PDF文件路径列表
        """
        if provider_strategy not in {
            "paper_scraper",
            "pypaperbot_scholar",
            "semantic_scholar_then_openalex",
            "semantic_scholar",
            "openalex",
        }:
            raise ValueError(
                "provider_strategy 必须是 pypaperbot_scholar、paper_scraper、"
                "semantic_scholar_then_openalex、semantic_scholar 或 openalex"
            )
        if provider_strategy != "paper_scraper":
            if not self._paper_search_execution_lock.acquire(blocking=False):
                raise RuntimeError("PaperSearch provider 已在另一个线程运行；拒绝并发 Scholar 请求")
            try:
                return self._search_papers_with_provider_strategy(
                    keywords=keywords,
                    pdf_dir=pdf_dir,
                    limit_per_keyword=limit_per_keyword,
                    status_sink=status_sink,
                    timeout_seconds=timeout_seconds,
                    target_papers=target_papers,
                    keyword_delay_seconds=keyword_delay_seconds,
                    candidate_page_size=candidate_page_size,
                    provider_strategy=provider_strategy,
                    ranking_question=ranking_question,
                    openalex_per_page=openalex_per_page,
                    download_delay_seconds=download_delay_seconds,
                    scholar_max_pages=scholar_max_pages,
                    scholar_results_per_page=scholar_results_per_page,
                    scholar_crossref_max_candidates=scholar_crossref_max_candidates,
                    scholar_request_delay_seconds=scholar_request_delay_seconds,
                    scholar_request_jitter_seconds=scholar_request_jitter_seconds,
                    scholar_initial_delay_seconds=scholar_initial_delay_seconds,
                    scholar_initial_jitter_seconds=scholar_initial_jitter_seconds,
                    scholar_backoff_base_seconds=scholar_backoff_base_seconds,
                    scholar_max_backoff_seconds=scholar_max_backoff_seconds,
                    scholar_max_consecutive_rate_limits=scholar_max_consecutive_rate_limits,
                    keyword_jitter_seconds=keyword_jitter_seconds,
                    resume_state_path=resume_state_path,
                    response_cache_dir=response_cache_dir,
                )
            finally:
                self._paper_search_execution_lock.release()

        if not PAPERSCRAPER_AVAILABLE:
            raise RuntimeError(
                "论文检索需要 paper-scraper（import paperscraper），请先按 README 安装"
            )

        keywords = list(dict.fromkeys(keywords))
        if not keywords:
            raise ValueError("论文检索至少需要一个关键词")
        if (
            not isinstance(limit_per_keyword, int)
            or isinstance(limit_per_keyword, bool)
            or limit_per_keyword <= 0
        ):
            raise ValueError("每关键词论文目标必须为正整数")
        if (
            not isinstance(candidate_page_size, int)
            or isinstance(candidate_page_size, bool)
            or candidate_page_size <= 0
        ):
            raise ValueError("候选页大小必须为正整数")
        os.makedirs(pdf_dir, exist_ok=True)
        if timeout_seconds is None:
            try:
                timeout_seconds = float(
                    os.environ.get("ARCHE_PAPERSEARCH_TIMEOUT_SECONDS", "180")
                )
            except ValueError:
                timeout_seconds = 180.0
        if keyword_delay_seconds is None:
            try:
                keyword_delay_seconds = float(
                    os.environ.get("ARCHE_PAPERSEARCH_KEYWORD_DELAY_SECONDS", "45")
                )
            except ValueError:
                keyword_delay_seconds = 45.0
        keyword_limits = (
            allocate_fair_paper_search_limits(keywords, target_papers)
            if target_papers is not None
            else {keyword: limit_per_keyword for keyword in keywords}
        )
        known_pdf_paths = set(self._collect_valid_pdf_paths(pdf_dir))

        with _enable_paperscraper_proxy_from_environment():
            for keyword in keywords:
                requested_limit = keyword_limits.get(keyword, 0)
                status: Dict[str, Any] = {
                    "keyword": keyword,
                    "status": "failed",
                    "downloaded_files": [],
                    "requested_limit": requested_limit,
                }
                if target_papers is not None and len(known_pdf_paths) >= target_papers:
                    status["status"] = "skipped_target_reached"
                    if status_sink is not None:
                        status_sink(dict(status))
                    continue
                if requested_limit <= 0:
                    status["status"] = "skipped_no_allocation"
                    if status_sink is not None:
                        status_sink(dict(status))
                    continue
                try:
                    logger.info(f"检索论文: {keyword}")
                    with _limit_single_keyword_paper_search_time(timeout_seconds):
                        papers = paperscraper.search_papers(
                            keyword,
                            limit=requested_limit,
                            pdir=pdf_dir,
                            # paperscraper 默认会递归翻页直到凑够可下载 PDF；
                            # _limit 只限制每页候选数，墙钟上限才是最终停止边界。
                            _limit=candidate_page_size,
                        )

                    if not isinstance(papers, (Mapping, list, tuple, set)):
                        raise TypeError(
                            "paperscraper.search_papers 返回值必须是路径映射或路径序列"
                        )
                    self._remove_empty_pdfs(pdf_dir)
                    valid_pdf_paths = set(self._collect_valid_pdf_paths(pdf_dir))
                    new_valid_paths = sorted(valid_pdf_paths - known_pdf_paths)
                    known_pdf_paths = valid_pdf_paths
                    status["status"] = "success"
                    status["downloaded_files"] = new_valid_paths
                    logger.info(f"为关键词 '{keyword}' 新增有效 PDF {len(new_valid_paths)} 篇")

                except Exception as e:
                    logger.error(f"检索关键词 '{keyword}' 失败: {e}")
                    status["error"] = str(e)
                    if isinstance(e, PaperSearchTimeoutError):
                        status["status"] = "timeout"
                        status["timeout_seconds"] = timeout_seconds
                finally:
                    if status_sink is not None:
                        status_sink(dict(status))
                    if keyword_delay_seconds and keyword_delay_seconds > 0:
                        time.sleep(keyword_delay_seconds)

        # 搜索函数可能在写完 PDF 后因超时才抛出异常，最终以目录扫描为准。
        self._remove_empty_pdfs(pdf_dir)
        return sorted(self._collect_valid_pdf_paths(pdf_dir))

    def run_paper_search_only(
        self,
        question: str,
        pdf_dir: str,
        manifest_path: str,
        candidate_page_size: int = 100,
        timeout_seconds: Optional[float] = 180.0,
        target_papers: int = 50,
        keyword_delay_seconds: float = 45.0,
        keywords_override: Optional[List[str]] = None,
        provider_strategy: str = "pypaperbot_scholar",
        openalex_per_page: int = 100,
        download_delay_seconds: float = 1.0,
        scholar_max_pages: int = 3,
        scholar_results_per_page: int = 10,
        scholar_crossref_max_candidates: int = 3,
        scholar_request_delay_seconds: Optional[float] = None,
        scholar_request_jitter_seconds: Optional[float] = None,
        scholar_initial_delay_seconds: Optional[float] = None,
        scholar_initial_jitter_seconds: Optional[float] = None,
        scholar_backoff_base_seconds: Optional[float] = None,
        scholar_max_backoff_seconds: Optional[float] = None,
        scholar_max_consecutive_rate_limits: Optional[int] = None,
        keyword_jitter_seconds: Optional[float] = None,
        resume_state_path: Optional[str] = None,
        response_cache_dir: Optional[str] = None,
        *,
        receipt_sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """执行关键词抽取和论文下载，并在 PaperSearch 后立即停止。

        输入：原始科学问题、PDF目录和 manifest 路径。动作：调用关键词模型，再按
        provider 策略调用 Google Scholar 或显式兼容 provider；随后只写脱敏的检索记录。副作用：创建 PDF目录和
        manifest 文件。返回：包含问题、关键词、逐关键词结果和成功 PDF 路径的记录。
        失败：关键词协议或论文依赖失败会抛出异常；单个关键词失败会记录在 manifest
        后继续其它关键词。此入口不构建 FAISS、不执行问答或文献综述。
        """

        if not isinstance(question, str) or not question.strip():
            raise ValueError("PaperSearch 问题不能为空")
        if not isinstance(manifest_path, str) or not manifest_path.strip():
            raise ValueError("PaperSearch manifest 路径不能为空")
        keyword_results: List[Dict[str, Any]] = []
        keywords = (
            normalize_frozen_paper_search_keywords(list(keywords_override))
            if keywords_override is not None
            else normalize_frozen_paper_search_keywords(
                self.extract_keywords(question, receipt_sink=receipt_sink)
            )
        )
        logger.info("PaperSearch 关键词冻结（来源=%s）：%s", "override" if keywords_override is not None else "model", keywords)
        if timeout_seconds is None:
            try:
                timeout_seconds = float(
                    os.environ.get("ARCHE_PAPERSEARCH_TIMEOUT_SECONDS", "180")
                )
            except ValueError:
                timeout_seconds = 180.0
        if target_papers <= 0:
            raise ValueError("全局论文目标必须为正整数")
        if keyword_delay_seconds < 0:
            raise ValueError("关键词退避秒数不能为负数")
        if (
            not isinstance(candidate_page_size, int)
            or isinstance(candidate_page_size, bool)
            or candidate_page_size <= 0
        ):
            raise ValueError("候选页大小必须为正整数")
        if (
            not isinstance(scholar_max_pages, int)
            or isinstance(scholar_max_pages, bool)
            or not 1 <= scholar_max_pages <= 20
        ):
            raise ValueError("Google Scholar 最大页数必须在1到20之间")
        if (
            not isinstance(scholar_results_per_page, int)
            or isinstance(scholar_results_per_page, bool)
            or not 1 <= scholar_results_per_page <= 20
        ):
            raise ValueError("Google Scholar 每页结果数必须在1到20之间")
        if (
            not isinstance(scholar_crossref_max_candidates, int)
            or isinstance(scholar_crossref_max_candidates, bool)
            or not 0 <= scholar_crossref_max_candidates <= 20
        ):
            raise ValueError("Scholar 每页 Crossref 补充上限必须在0到20之间")
        self.search_papers(
            keywords,
            pdf_dir,
            status_sink=keyword_results.append,
            timeout_seconds=timeout_seconds,
            target_papers=target_papers,
            keyword_delay_seconds=keyword_delay_seconds,
            candidate_page_size=candidate_page_size,
            provider_strategy=provider_strategy,
            ranking_question=question,
            openalex_per_page=openalex_per_page,
            download_delay_seconds=download_delay_seconds,
            scholar_max_pages=scholar_max_pages,
            scholar_results_per_page=scholar_results_per_page,
            scholar_crossref_max_candidates=scholar_crossref_max_candidates,
            scholar_request_delay_seconds=scholar_request_delay_seconds,
            scholar_request_jitter_seconds=scholar_request_jitter_seconds,
            scholar_initial_delay_seconds=scholar_initial_delay_seconds,
            scholar_initial_jitter_seconds=scholar_initial_jitter_seconds,
            scholar_backoff_base_seconds=scholar_backoff_base_seconds,
            scholar_max_backoff_seconds=scholar_max_backoff_seconds,
            scholar_max_consecutive_rate_limits=scholar_max_consecutive_rate_limits,
            keyword_jitter_seconds=keyword_jitter_seconds,
            resume_state_path=resume_state_path,
            response_cache_dir=response_cache_dir,
        )
        existing_files = self._collect_unique_pdf_paths_by_sha256(pdf_dir)
        target_reached = len(existing_files) >= target_papers

        try:
            installed_paper_scraper_version = importlib_metadata.version("paper-scraper")
        except importlib_metadata.PackageNotFoundError:
            installed_paper_scraper_version = "unknown"
        search_config = {
            "candidate_page_size": candidate_page_size,
            "timeout_seconds": timeout_seconds,
            "target_papers": target_papers,
            "keyword_delay_seconds": float(keyword_delay_seconds),
            "keyword_allocations": self._paper_search_audit.get(
                "keyword_allocations", {}
            ),
            "keyword_allocations_applied": self._paper_search_audit.get(
                "keyword_allocations_applied", True
            ),
            "keyword_allocation_history": self._paper_search_audit.get(
                "keyword_allocation_history", []
            ),
            "allocation_strategy": "dynamic_remaining_gap_fair_allocation",
            "pdf_dir": os.path.abspath(pdf_dir),
            "provider_strategy": provider_strategy,
            "openalex_per_page": openalex_per_page,
            "download_delay_seconds": download_delay_seconds,
            "scholar_max_pages": scholar_max_pages,
            "scholar_results_per_page": scholar_results_per_page,
            "scholar_crossref_max_candidates": scholar_crossref_max_candidates,
            **self._paper_search_audit.get("effective_pacing", {}),
            "resume_state_path": os.path.abspath(resume_state_path)
            if resume_state_path
            else os.path.abspath(os.path.join(pdf_dir, ".paper-search-resume.json")),
            "response_cache_dir": os.path.abspath(response_cache_dir)
            if response_cache_dir
            else os.path.abspath(os.path.join(pdf_dir, ".paper-search-cache")),
            "ranking_strategy": "local_bge_title_abstract_similarity",
            "provider": {
                "name": provider_strategy,
                "primary_metadata_api": (
                    "google_scholar_html"
                    if provider_strategy == "pypaperbot_scholar"
                    else "semantic_scholar_graph_v1"
                ),
                "fallback_metadata_api": (
                    "none_for_google_scholar"
                    if provider_strategy == "pypaperbot_scholar"
                    else "openalex_works_api"
                ),
                "public_pdf_policy": "scholar_explicit_http_url_only",
                "shadow_library_policy": "blocked",
                "scholar_strategy_version": "pypaperbot_scholar_v1",
                "crossref_metadata_api": (
                    "crossref_works_title_lookup"
                    if provider_strategy == "pypaperbot_scholar"
                    and scholar_crossref_max_candidates > 0
                    else "disabled"
                ),
                "legacy_compatibility_distribution": "paper-scraper",
                "legacy_compatibility_source_tag": "v1.8.1",
                "legacy_compatibility_installed_version": installed_paper_scraper_version,
            },
        }

        manifest = {
            "schema_version": 1,
            "status": "completed" if target_reached else "partial",
            "stopped_after": "paper_search",
            "question": question,
            "keywords": keywords,
            "search_config": search_config,
            "keyword_results": keyword_results,
            "keyword_source": "override" if keywords_override is not None else "model",
            "downloaded_files": list(dict.fromkeys(existing_files)),
            "downloaded_pdf_count": len(set(existing_files)),
            "target_papers": target_papers,
            "actual_papers": len(set(existing_files)),
            "target_reached": target_reached,
            "corpus_fingerprint": self._calculate_pdf_corpus_fingerprint(pdf_dir),
            "corpus_files": self._collect_pdf_corpus_records(pdf_dir),
            "index_built": False,
            "downstream_stages_run": [],
            "provider_attempts": self._paper_search_audit.get("provider_attempts", []),
            "download_attempts": self._paper_search_audit.get("download_attempts", []),
            "accepted_papers": self._paper_search_audit.get("accepted_papers", []),
            "statistics": {
                "candidate_count": self._paper_search_audit.get("candidate_count", 0),
                "download_count": self._paper_search_audit.get("download_count", 0),
                "deduplicated_count": self._paper_search_audit.get("deduplicated_count", 0),
                "failed_download_count": self._paper_search_audit.get("failed_download_count", 0),
                "target_stop_reached": self._paper_search_audit.get("target_stop_reached", False),
            },
        }
        manifest_parent = os.path.dirname(os.path.abspath(manifest_path))
        if manifest_parent:
            os.makedirs(manifest_parent, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        logger.info(
            "PaperSearch-only 完成：关键词=%d，成功 PDF=%d；未构建索引或运行后续阶段",
            len(keywords),
            manifest["downloaded_pdf_count"],
        )
        return manifest

    def _collect_valid_pdf_paths(self, folder_path: str) -> List[str]:
        """扫描目录并返回可由 PyMuPDF 打开的非空 PDF 绝对路径。"""

        if not os.path.isdir(folder_path):
            return []
        valid_paths = []
        for filename in sorted(os.listdir(folder_path)):
            if not filename.lower().endswith(".pdf"):
                continue
            file_path = os.path.abspath(os.path.join(folder_path, filename))
            try:
                if os.path.getsize(file_path) <= 0:
                    continue
                with fitz.open(file_path) as document:
                    if document.page_count > 0:
                        valid_paths.append(file_path)
            except Exception as exc:
                logger.warning("忽略不可读 PDF %s：%s", file_path, exc)
        return valid_paths

    def _collect_unique_pdf_paths_by_sha256(self, folder_path: str) -> List[str]:
        """扫描可读 PDF，并按完整内容 SHA-256 只保留一份路径。"""

        unique_paths: Dict[str, str] = {}
        for file_path in self._collect_valid_pdf_paths(folder_path):
            try:
                with open(file_path, "rb") as handle:
                    digest = hashlib.sha256(handle.read()).hexdigest()
            except OSError:
                continue
            unique_paths.setdefault(digest, file_path)
        return sorted(unique_paths.values())

    def _calculate_pdf_corpus_fingerprint(self, folder_path: str) -> str:
        """按 PDF 相对路径和内容 SHA-256 计算语料指纹。"""

        records = self._collect_pdf_corpus_records(folder_path)
        payload = json.dumps(
            records,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _collect_pdf_corpus_records(self, folder_path: str) -> List[Dict[str, Any]]:
        """收集按相对 POSIX 路径排序的 PDF 内容身份。"""

        records: List[Dict[str, Any]] = []
        root = os.path.abspath(folder_path)
        paths = self._collect_unique_pdf_paths_by_sha256(root)
        for path in paths:
            try:
                with open(path, "rb") as handle:
                    content = handle.read()
            except OSError:
                continue
            records.append(
                {
                    "path": PurePosixPath(os.path.relpath(path, root)).as_posix(),
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "size": len(content),
                }
            )
        return sorted(records, key=lambda item: item["path"])

    def _embedding_model_identity(self) -> Dict[str, Any]:
        """返回不包含绝对路径和时间戳的本地 BGE 模型身份。"""

        model_path = self.model_path
        if isinstance(model_path, str) and model_path:
            root = os.path.abspath(os.path.expanduser(model_path))
            files: List[Dict[str, Any]] = []
            if os.path.isdir(root):
                for current_root, _dirs, names in os.walk(root):
                    for name in sorted(names):
                        path = os.path.join(current_root, name)
                        if not os.path.isfile(path):
                            continue
                        try:
                            with open(path, "rb") as handle:
                                content = handle.read()
                        except OSError:
                            continue
                        files.append(
                            {
                                "path": PurePosixPath(os.path.relpath(path, root)).as_posix(),
                                "sha256": hashlib.sha256(content).hexdigest(),
                                "size": len(content),
                            }
                        )
            payload = {"kind": "local_sentence_transformer", "files": sorted(files, key=lambda item: item["path"])}
        else:
            encoder_type = type(self.embedder)
            payload = {
                "kind": "injected_sentence_transformer",
                "class": f"{encoder_type.__module__}.{encoder_type.__qualname__}",
            }
            for name in ("model_name", "model_id", "model_fingerprint"):
                value = getattr(self.embedder, name, None)
                if isinstance(value, (str, int, float, bool)):
                    payload[name] = value
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return {
            "fingerprint": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            "descriptor": payload,
        }

    def _embedding_dimension(self) -> int:
        """读取当前 BGE 编码器维度，无法确认时通过一次本地探针取得。"""

        for name in ("get_sentence_embedding_dimension",):
            getter = getattr(self.embedder, name, None)
            if callable(getter):
                value = getter()
                if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                    return value
        for name in ("dimension", "embedding_dimension"):
            value = getattr(self.embedder, name, None)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
        vectors = self._get_embeddings(["__arche_index_dimension_probe__"])
        return int(vectors.shape[1])

    @staticmethod
    def _write_json_atomically(path: str, payload: Mapping[str, Any]) -> None:
        """原子写入 JSON，避免中断留下半份 manifest 或索引元数据。"""

        destination = os.path.abspath(path)
        parent = os.path.dirname(destination)
        os.makedirs(parent, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(destination)}.", suffix=".part", dir=parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                try:
                    os.unlink(temporary)
                except OSError:
                    pass

    @staticmethod
    def _sanitize_manifest_value(value: Any, pdf_dir: str, *, key: str = "") -> Any:
        """把 manifest 中的本地路径转换为相对路径或稳定的逻辑引用。"""

        root = os.path.abspath(pdf_dir)
        path_keys = {
            "path",
            "pdf_dir",
            "manifest_path",
            "resume_state_path",
            "response_cache_dir",
            "downloaded_files",
        }
        if isinstance(value, Mapping):
            return {
                str(item_key): RetrievalAgent._sanitize_manifest_value(
                    item_value, root, key=str(item_key)
                )
                for item_key, item_value in value.items()
            }
        if isinstance(value, list):
            return [
                RetrievalAgent._sanitize_manifest_value(item, root, key=key)
                for item in value
            ]
        if isinstance(value, tuple):
            return [
                RetrievalAgent._sanitize_manifest_value(item, root, key=key)
                for item in value
            ]
        if isinstance(value, str) and key in path_keys:
            candidate = os.path.expanduser(value)
            if os.path.isabs(candidate):
                try:
                    relative = os.path.relpath(os.path.abspath(candidate), root)
                except ValueError:
                    relative = ""
                if relative == ".":
                    return "."
                if relative and relative != os.pardir and not relative.startswith(os.pardir + os.sep):
                    return PurePosixPath(relative).as_posix()
                digest = hashlib.sha256(os.path.abspath(candidate).encode("utf-8")).hexdigest()
                return f"external:{digest}"
            return PurePosixPath(candidate).as_posix()
        return value

    def _write_paper_search_manifest(
        self,
        *,
        question: str,
        keywords: List[str],
        pdf_dir: str,
        manifest_path: str,
        target_papers: int,
        search_config: Mapping[str, Any],
        keyword_results: Sequence[Mapping[str, Any]],
        keyword_source: str = "model",
    ) -> Dict[str, Any]:
        """写入正式 PaperSearch manifest，并返回同一份脱敏事实。"""

        existing_files = self._collect_unique_pdf_paths_by_sha256(pdf_dir)
        corpus_files = self._collect_pdf_corpus_records(pdf_dir)
        corpus_fingerprint = self._calculate_pdf_corpus_fingerprint(pdf_dir)
        target_reached = len(existing_files) >= target_papers
        sanitized_config = self._sanitize_manifest_value(dict(search_config), pdf_dir)
        sanitized_config.pop("pdf_dir", None)
        sanitized_keyword_results = self._sanitize_manifest_value(
            [dict(item) for item in keyword_results], pdf_dir
        )
        sanitized_downloaded_files = [
            item["path"] for item in corpus_files
        ]
        sanitized_accepted = self._sanitize_manifest_value(
            self._paper_search_audit.get("accepted_papers", []), pdf_dir
        )
        accepted_sha256 = {
            item.get("sha256")
            for item in sanitized_accepted
            if isinstance(item, Mapping) and isinstance(item.get("sha256"), str)
        }
        for record in corpus_files:
            if record["sha256"] in accepted_sha256:
                continue
            sanitized_accepted.append(
                {
                    "path": record["path"],
                    "sha256": record["sha256"],
                    "size": record["size"],
                    "provider": "resume_existing",
                    "keyword": None,
                    "identifiers": [],
                    "source_host": "",
                }
            )
        sanitized_accepted = sorted(
            sanitized_accepted,
            key=lambda item: str(item.get("sha256", ""))
            if isinstance(item, Mapping)
            else str(item),
        )
        manifest = {
            "schema_version": 2,
            "status": "completed" if target_reached else "partial",
            "stopped_after": "paper_search",
            "question": question,
            "keywords": list(keywords),
            "search_config": sanitized_config,
            "keyword_results": sanitized_keyword_results,
            "keyword_source": keyword_source,
            "downloaded_files": sanitized_downloaded_files,
            "downloaded_pdf_count": len(existing_files),
            "target_papers": target_papers,
            "actual_papers": len(existing_files),
            "target_reached": target_reached,
            "index_built": False,
            "downstream_stages_run": [],
            "provider_attempts": self._paper_search_audit.get("provider_attempts", []),
            "download_attempts": self._paper_search_audit.get("download_attempts", []),
            "accepted_papers": sanitized_accepted,
            "corpus_files": corpus_files,
            "corpus_fingerprint": corpus_fingerprint,
            "keyword_allocations": self._paper_search_audit.get("keyword_allocations", {}),
            "keyword_allocation_history": self._paper_search_audit.get(
                "keyword_allocation_history", []
            ),
            "keyword_allocations_applied": self._paper_search_audit.get(
                "keyword_allocations_applied", True
            ),
            "global_target_cap": self._paper_search_audit.get(
                "global_target_cap", target_papers
            ),
            "statistics": {
                "candidate_count": self._paper_search_audit.get("candidate_count", 0),
                "download_count": self._paper_search_audit.get("download_count", 0),
                "deduplicated_count": self._paper_search_audit.get("deduplicated_count", 0),
                "failed_download_count": self._paper_search_audit.get("failed_download_count", 0),
                "target_stop_reached": self._paper_search_audit.get("target_stop_reached", False),
            },
        }
        self._write_json_atomically(manifest_path, manifest)
        return manifest

    @staticmethod
    def _validate_frozen_paper_manifest(
        manifest_path: str,
        pdf_dir: str,
        *,
        target_papers: Optional[int] = None,
    ) -> Dict[str, Any]:
        """校验 no-search 复用的 manifest、目标数量和磁盘 PDF 内容。"""

        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, ValueError) as exc:
            raise ValueError(f"PaperSearch manifest 无法读取：{manifest_path}") from exc
        if not isinstance(manifest, Mapping):
            raise ValueError("PaperSearch manifest 必须是 object")
        schema_version = manifest.get("schema_version")
        if schema_version not in {1, 2}:
            raise ValueError("PaperSearch manifest schema_version 不受支持")
        target = manifest.get("target_papers")
        actual = manifest.get("actual_papers")
        if target_papers is not None and target != target_papers:
            raise ValueError("PaperSearch manifest 目标数量与 paper_search 配置不一致")
        if (
            manifest.get("status") != "completed"
            or manifest.get("target_reached") is not True
            or not isinstance(target, int)
            or isinstance(target, bool)
            or target <= 0
            or not isinstance(actual, int)
            or isinstance(actual, bool)
            or actual < target
        ):
            raise ValueError("PaperSearch manifest 尚未冻结或未达到目标数量")
        paths = RetrievalAgent._collect_unique_pdf_paths_for_manifest(pdf_dir)
        if len(paths) != actual or len(paths) < target:
            raise ValueError("PaperSearch manifest 与磁盘唯一 PDF 数量不一致")
        disk_sha256 = set()
        for path in paths:
            with open(path, "rb") as handle:
                disk_sha256.add(hashlib.sha256(handle.read()).hexdigest())
        accepted_papers = manifest.get("accepted_papers")
        if not isinstance(accepted_papers, list):
            raise ValueError("PaperSearch manifest 缺少 accepted_papers")
        accepted_sha256 = {
            item.get("sha256")
            for item in accepted_papers
            if isinstance(item, Mapping) and isinstance(item.get("sha256"), str)
        }
        if accepted_sha256 != disk_sha256:
            raise ValueError("PaperSearch manifest accepted_papers 与磁盘 PDF SHA-256 集不一致")
        manifest_fingerprint = manifest.get("corpus_fingerprint")
        if schema_version == 2 and not isinstance(manifest_fingerprint, str):
            raise ValueError("PaperSearch v2 manifest 缺少 corpus_fingerprint")
        if isinstance(manifest_fingerprint, str) and manifest_fingerprint:
            root = os.path.abspath(pdf_dir)
            records = []
            for filename in (sorted(os.listdir(root)) if os.path.isdir(root) else []):
                if not filename.lower().endswith(".pdf"):
                    continue
                path = os.path.abspath(os.path.join(root, filename))
                try:
                    if os.path.getsize(path) <= 0:
                        continue
                    with fitz.open(path) as document:
                        if document.page_count <= 0:
                            continue
                    with open(path, "rb") as handle:
                        content = handle.read()
                except (OSError, RuntimeError, ValueError):
                    continue
                records.append(
                    {
                        "path": PurePosixPath(os.path.relpath(path, root)).as_posix(),
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "size": len(content),
                    }
                )
            records = sorted(records, key=lambda item: item["path"])
            current = hashlib.sha256(
                json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if current != manifest_fingerprint:
                raise ValueError("PaperSearch manifest 与当前 PDF 语料指纹不一致")
            stored_records = manifest.get("corpus_files")
            if schema_version == 2 and stored_records != records:
                raise ValueError("PaperSearch manifest corpus_files 与当前 PDF 语料不一致")
        return dict(manifest)

    @staticmethod
    def _collect_unique_pdf_paths_for_manifest(folder_path: str) -> List[str]:
        """无实例地收集可读且按 SHA-256 去重的 PDF 路径。"""

        if not os.path.isdir(folder_path):
            return []
        unique: Dict[str, str] = {}
        for filename in sorted(os.listdir(folder_path)):
            if not filename.lower().endswith(".pdf"):
                continue
            path = os.path.abspath(os.path.join(folder_path, filename))
            try:
                if os.path.getsize(path) <= 0:
                    continue
                with fitz.open(path) as document:
                    if document.page_count <= 0:
                        continue
                with open(path, "rb") as handle:
                    unique.setdefault(hashlib.sha256(handle.read()).hexdigest(), path)
            except (OSError, RuntimeError, ValueError):
                continue
        return sorted(unique.values())

    def _remove_empty_pdfs(self, folder_path: str):
        """删除空文件或无页面 PDF，保留可供后续索引的文件。"""
        if not os.path.exists(folder_path):
            return

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(folder_path, filename)
                try:
                    # 检查文件大小
                    if os.path.getsize(file_path) == 0:
                        logger.info(f"删除空文件: {file_path}")
                        os.remove(file_path)
                        continue

                    # 尝试打开PDF
                    doc = fitz.open(file_path)
                    if doc.page_count == 0:
                        logger.info(f"删除空PDF（无页）: {file_path}")
                        doc.close()
                        os.remove(file_path)
                    else:
                        doc.close()

                except Exception as e:
                    logger.warning(f"检查PDF时出错 {file_path}: {e}")

    # ==================== 文本处理 ====================

    def _load_pdf_text(self, pdf_path: str) -> List[Tuple[str, int]]:
        """从PDF加载文本（按段落）"""
        try:
            doc = fitz.open(pdf_path)
            paragraphs = []

            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text")
                if text.strip():
                    # 按空行分段
                    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
                    for para in paras:
                        paragraphs.append((para, page_num))

            doc.close()
            return paragraphs

        except Exception as e:
            logger.error(f"读取PDF失败 {pdf_path}: {e}")
            return []

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """文本分块"""
        words = text.split()
        chunks = []
        start = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += chunk_size - overlap

        return chunks

    # ==================== 向量嵌入 ====================

    def _get_embeddings(
        self,
        texts: List[str],
    ) -> np.ndarray:
        """使用 SentenceTransformer 生成归一化语义嵌入，失败时直接终止。

        输入：待编码的文本列表。动作：调用本地 BGE 编码器并校验形状、有限值和
        向量范数。副作用：可能触发模型首次加载后的 CPU 推理。返回：可直接用于
        FAISS 内积检索的 ``float32`` 矩阵。失败：编码器缺失、编码异常或返回非法
        数组都会抛出 ``RetrievalEmbeddingError``，不会降级为词频匹配。
        """

        if self.embedder is None:
            raise RetrievalEmbeddingError("SentenceTransformer 编码器未初始化")
        if not texts:
            return np.empty((0, 0), dtype="float32")
        try:
            encoded = self.embedder.encode(texts, normalize_embeddings=True)
            vectors = np.asarray(encoded, dtype="float32")
        except PaperSearchTimeoutError:
            raise
        except Exception as exc:
            raise RetrievalEmbeddingError(
                f"SentenceTransformer 语义嵌入生成失败：{exc}"
            ) from exc
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts) or vectors.shape[1] == 0:
            raise RetrievalEmbeddingError(
                "SentenceTransformer 返回的嵌入形状非法："
                f"shape={vectors.shape!r}，文本数={len(texts)}"
            )
        if not np.isfinite(vectors).all():
            raise RetrievalEmbeddingError("SentenceTransformer 返回了非有限嵌入值")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise RetrievalEmbeddingError("SentenceTransformer 返回了零范数嵌入")
        return (vectors / norms).astype("float32")

    # ==================== 索引构建 ====================

    def build_index(self, pdf_dir: str, index_dir: str):
        """
        构建论文索引

        参数:
            pdf_dir: PDF目录
            index_dir: 索引存储目录
        """
        # 关闭在线搜索时，默认论文目录可能尚未创建。空目录代表“当前没有本地论文”，
        # 不应被误报成检索技术失败；后续仍会生成问题自身的化学上下文。
        os.makedirs(pdf_dir, exist_ok=True)
        os.makedirs(index_dir, exist_ok=True)

        self.index = None
        self.documents = []
        self.metadatas = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        # 加载所有PDF文本
        for pdf_path in self._collect_unique_pdf_paths_by_sha256(pdf_dir):
            filename = os.path.basename(pdf_path)
            paragraphs = self._load_pdf_text(pdf_path)

            for text, page_num in paragraphs:
                chunks = self._chunk_text(text)

                for chunk in chunks:
                    # 检测章节类型
                    section_type = "methods" if re.search(
                        r"computational|method|calculation|theory",
                        chunk, re.I
                    ) else "general"

                    documents.append(chunk)
                    metadatas.append({
                        "source": filename,
                        "page": page_num,
                        "section": section_type
                    })

        if not documents:
            logger.warning("没有找到可索引的文档")
            return

        # 生成嵌入向量
        logger.info(f"为 {len(documents)} 个文档块生成嵌入...")
        vectors = self._get_embeddings(
            documents,
        )

        # 构建FAISS索引
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        corpus_records = self._collect_pdf_corpus_records(pdf_dir)
        corpus_fingerprint = self._calculate_pdf_corpus_fingerprint(pdf_dir)
        model_identity = self._embedding_model_identity()
        generation_id = str(uuid.uuid4())
        metadata = {
            "schema_version": 2,
            "generation_id": generation_id,
            "documents": documents,
            "metadatas": metadatas,
            "embedder_name": "bge",
            "embedding_backend": "sentence-transformers",
            "embedding_model_identity": model_identity,
            "embedding_dimension": dim,
            "embedding_vector_dimension": int(vectors.shape[1]),
            "corpus_files": corpus_records,
            "corpus_fingerprint": corpus_fingerprint,
            "document_count": len(documents),
            "documents_fingerprint": hashlib.sha256(
                json.dumps(
                    {"documents": documents, "metadatas": metadatas},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "chunk_algorithm": "word_window",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "text_extraction_version": "fitz_text_v1",
            "faiss_index_type": "IndexFlatIP",
            "faiss_metric": "inner_product",
            "normalized_embeddings": True,
        }
        index_path = os.path.join(index_dir, "index.faiss")
        docs_path = os.path.join(index_dir, "docs.json")
        index_tmp = f"{index_path}.{generation_id}.part"
        docs_tmp = f"{docs_path}.{generation_id}.part"
        faiss.write_index(index, index_tmp)
        with open(index_tmp, "rb") as handle:
            index_bytes = handle.read()
        with open(index_tmp, "ab") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        with open(docs_tmp, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        with open(docs_tmp, "rb") as handle:
            docs_bytes = handle.read()
        os.replace(index_tmp, index_path)
        os.replace(docs_tmp, docs_path)
        marker = {
            "schema_version": 2,
            "committed": True,
            "generation_id": generation_id,
            "index_file": "index.faiss",
            "docs_file": "docs.json",
            "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
            "index_size": len(index_bytes),
            "docs_sha256": hashlib.sha256(docs_bytes).hexdigest(),
            "docs_size": len(docs_bytes),
        }
        self._write_json_atomically(os.path.join(index_dir, "index.commit.json"), marker)

        self.index = index
        self.documents = documents
        self.metadatas = metadatas

        logger.info(f"索引构建完成: {len(documents)} 个文档块")

    def load_index(self, index_dir: str, *, pdf_dir: Optional[str] = None):
        """加载 BGE 索引，并在提供 PDF 目录时核对语料指纹。"""
        self.index = None
        self.documents = []
        self.metadatas = []
        if pdf_dir is None:
            logger.warning("加载正式索引必须提供 pdf_dir 以核对语料指纹")
            return False
        try:
            index_path = os.path.join(index_dir, "index.faiss")
            docs_path = os.path.join(index_dir, "docs.json")
            marker_path = os.path.join(index_dir, "index.commit.json")
            if not all(os.path.isfile(path) for path in (index_path, docs_path, marker_path)):
                logger.warning("索引提交标记或文件缺失，要求重建")
                return False
            with open(marker_path, "r", encoding="utf-8") as handle:
                marker = json.load(handle)
            if (
                marker.get("schema_version") != 2
                or marker.get("index_file") != "index.faiss"
                or marker.get("docs_file") != "docs.json"
                or not isinstance(marker.get("generation_id"), str)
                or marker.get("committed") is not True
            ):
                return False
            with open(index_path, "rb") as handle:
                index_bytes = handle.read()
            with open(docs_path, "rb") as handle:
                docs_bytes = handle.read()
            if (
                marker.get("index_sha256") != hashlib.sha256(index_bytes).hexdigest()
                or marker.get("docs_sha256") != hashlib.sha256(docs_bytes).hexdigest()
                or marker.get("index_size") != len(index_bytes)
                or marker.get("docs_size") != len(docs_bytes)
            ):
                return False
            with open(docs_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("generation_id") != marker.get("generation_id"):
                return False
            documents = data.get("documents", [])
            metadatas = data.get("metadatas", [])
            if (
                not isinstance(documents, list)
                or not isinstance(metadatas, list)
                or not documents
                or len(documents) != len(metadatas)
                or not all(isinstance(item, str) and item for item in documents)
                or not all(isinstance(item, Mapping) for item in metadatas)
            ):
                return False

            stored_embedder_name = data.get("embedder_name")
            if (
                data.get("schema_version") != 2
                or stored_embedder_name != "bge"
                or data.get("embedding_backend") != "sentence-transformers"
                or data.get("chunk_algorithm") != "word_window"
                or data.get("chunk_size") != 500
                or data.get("chunk_overlap") != 50
                or data.get("text_extraction_version") != "fitz_text_v1"
                or data.get("faiss_index_type") != "IndexFlatIP"
                or data.get("faiss_metric") != "inner_product"
                or data.get("normalized_embeddings") is not True
            ):
                logger.error(
                    "拒绝加载非 SentenceTransformer 索引：embedder_name=%r",
                    stored_embedder_name,
                )
                return False
            if (
                not isinstance(data.get("embedding_dimension"), int)
                or isinstance(data["embedding_dimension"], bool)
                or data["embedding_dimension"] <= 0
                or data.get("document_count") != len(documents)
            ):
                return False
            loaded_index = faiss.read_index(index_path)
            if loaded_index.ntotal != len(documents) or len(documents) != len(metadatas):
                return False
            if loaded_index.d != data["embedding_dimension"] or loaded_index.__class__.__name__ != "IndexFlatIP":
                return False
            stored_fingerprint = data.get("corpus_fingerprint")
            if not isinstance(stored_fingerprint, str) or not stored_fingerprint:
                logger.warning("索引缺少 corpus_fingerprint，要求重建")
                return False
            current_records = self._collect_pdf_corpus_records(pdf_dir)
            current_fingerprint = self._calculate_pdf_corpus_fingerprint(pdf_dir)
            if stored_fingerprint != current_fingerprint or data.get("corpus_files") != current_records:
                logger.info("PDF 语料已变化，拒绝复用旧索引")
                return False
            stored_model_identity = data.get("embedding_model_identity")
            if not isinstance(stored_model_identity, Mapping):
                return False
            current_model_identity = self._embedding_model_identity()
            if stored_model_identity != current_model_identity:
                return False
            if data.get("embedding_vector_dimension") != self._embedding_dimension():
                return False
            documents_fingerprint = hashlib.sha256(
                json.dumps(
                    {"documents": documents, "metadatas": metadatas},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if data.get("documents_fingerprint") != documents_fingerprint:
                return False
            stored_dimension = data.get("embedding_dimension")
            if stored_dimension != loaded_index.d:
                logger.error(
                    "索引维度元数据不一致：metadata=%r，faiss=%r",
                    stored_dimension,
                    loaded_index.d,
                )
                return False

            self.index = loaded_index
            self.documents = documents
            self.metadatas = metadatas
            logger.info(f"索引加载成功: {len(documents)} 个文档块")
            return True

        except Exception as e:
            self.index = None
            self.documents = []
            self.metadatas = []
            logger.error(f"加载索引失败: {e}")
            return False

    # ==================== 检索查询 ====================

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        检索相关文档

        参数:
            query: 查询文本
            top_k: 返回结果数量

        返回:
            检索结果列表
        """
        if self.index is None or not self.documents:
            logger.warning("索引未加载，无法检索")
            return []

        # 获取查询向量
        query_vec = self._get_embeddings([query])

        # 搜索
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append({
                    "text": self.documents[idx],
                    "score": float(scores[0][i]),
                    "metadata": self.metadatas[idx]
                })

        # Methods部分优先排序
        results.sort(key=lambda x: 0 if x["metadata"]["section"] == "methods" else 1)

        return results

    # ==================== 问答系统 ====================

    @staticmethod
    def _build_model_documents(results: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        """把检索结果转换为文献模型的安全输入片段。

        输入：``search`` 返回的结果数组，其中 metadata 由本地索引保存 PDF 来源和
        1-based 页码。动作：保留正文、内部 PDF 身份和合法页码；内部身份仅供
        ``model_prompts`` 在一次请求内合并同一 PDF 的多个 chunk，绝不直接进入
        模型消息。副作用：不修改结果或索引。返回：可交给 QA/综述 projector 的
        文档数组。失败：非 object 结果被跳过，缺失正文按空字符串记录；非法页码
        不会被猜测或转换。
        """

        documents: List[Dict[str, Any]] = []
        for item in results:
            if not isinstance(item, Mapping):
                continue
            text = item.get("text", "")
            if not isinstance(text, str):
                text = str(text)
            document: Dict[str, Any] = {"text": text}
            metadata = item.get("metadata")
            if isinstance(metadata, Mapping):
                # source 只作为本次请求内的分组键。它可能是文件名或内部路径，
                # 由 projector 重新映射为 paper_N 后才允许进入模型消息。
                source = metadata.get("paper_key")
                if source is None:
                    source = metadata.get("source")
                if source is not None:
                    document["_paper_key"] = source
                page = metadata.get("page")
                if isinstance(page, int) and not isinstance(page, bool) and page >= 1:
                    document["page"] = page
            documents.append(document)
        return documents

    def answer_question(
        self,
        question: str,
        top_k: int = 7,
        *,
        receipt_sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
    ) -> str:
        """
        基于检索结果回答问题

        参数:
            question: 问题
            top_k: 使用的检索结果数量

        返回:
            答案文本
        """
        # 检索相关文档
        results = self.search(question, top_k)

        if not results:
            return ""
        documents = self._build_model_documents(results)
        return self._call_retrieval_catalog_stage(
            stage="document_qa",
            base_input={"question": question, "documents": documents},
            receipt_sink=receipt_sink,
        )

    def generate_literature_review(
        self,
        research_topic: str,
        top_k: int = 7,
        *,
        receipt_sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
    ) -> str:
        """
        生成文献综述

        参数:
            research_topic: 研究主题
            top_k: 使用的检索结果数量

        返回:
            文献综述文本
        """
        # 检索相关文档
        results = self.search(research_topic, top_k)

        if not results:
            return ""
        documents = self._build_model_documents(results)
        return self._call_retrieval_catalog_stage(
            stage="literature_review",
            base_input={
                "research_topic": research_topic,
                "documents": documents,
            },
            receipt_sink=receipt_sink,
        )

    # ==================== 结构化检索输出 ====================

    def _normalize_retrieval_result(self,
                                  question: str,
                                  keywords: List[str],
                                  retrieved_papers: List[str],
                                  answer: str,
                                  literature_review: str,
                                  index_built: bool) -> Dict[str, Any]:
        """
        规范化检索结果为结构化输出

        参数:
            question: 科学问题
            keywords: 关键词列表
            retrieved_papers: 检索到的论文列表
            answer: 答案文本
            literature_review: 文献综述
            index_built: 索引是否构建成功

        返回:
            结构化检索结果
        """
        # 提取机制线索
        mechanistic_clues = self._extract_mechanistic_clues(literature_review)

        # 识别局限性
        limitations = self._identify_limitations(literature_review, answer)

        # 结构化结果
        structured_result = {
            "question": question,
            "keywords": keywords,
            "retrieved_papers": retrieved_papers,
            "literature_review": literature_review,
            "answer": answer,
            "index_built": index_built,
            "mechanistic_clues": mechanistic_clues,
            "limitations": limitations,
            "retrieval_timestamp": time.time(),
            "structured_version": "1.0"
        }

        return structured_result

    def _extract_mechanistic_clues(self, literature_review: str) -> List[str]:
        """
        从文献综述中提取机制线索

        参数:
            literature_review: 文献综述文本

        返回:
            机制线索列表
        """
        clues = []

        if not literature_review:
            return clues

        # 查找机制相关的关键词
        mechanistic_patterns = [
            r"transition state(?:s)? (?:for|of|in) ([^.]*?)[.]",
            r"reaction mechanism (?:for|of|involving) ([^.]*?)[.]",
            r"catalytic cycle (?:for|of) ([^.]*?)[.]",
            r"rate-determining step(?:s)? (?:for|of) ([^.]*?)[.]",
            r"key intermediate(?:s)? (?:in|for) ([^.]*?)[.]",
            r"activation barrier(?:s)? (?:for|of) ([^.]*?)[.]",
            r"photochemical excitation (?:of|for) ([^.]*?)[.]",
            r"conformational change(?:s)? (?:in|for) ([^.]*?)[.]",
            r"solvent effect(?:s)? (?:on|in) ([^.]*?)[.]",
            r"stereoselectivity (?:of|in) ([^.]*?)[.]"
        ]

        for pattern in mechanistic_patterns:
            matches = re.findall(pattern, literature_review, re.IGNORECASE)
            for match in matches:
                clue = match.strip()
                if clue and clue not in clues:
                    clues.append(clue)

        # 如果没有找到模式，尝试提取包含关键词的句子
        if not clues:
            sentences = re.split(r"[.!?]", literature_review)
            for sentence in sentences:
                sentence = sentence.strip()
                if any(keyword in sentence.lower() for keyword in [
                    "mechanism", "transition state", "catalytic", "barrier",
                    "intermediate", "reaction pathway", "ts", "irc"
                ]):
                    if sentence and len(sentence) < 200:
                        clues.append(sentence)

        return clues[:10]  # 最多10个线索

    def _identify_limitations(self, literature_review: str, answer: str) -> List[str]:
        """从文献综述和答案中提取有明确文字证据的局限性。"""
        limitations = []

        combined_text = (literature_review + " " + answer).lower()

        # 查找局限性相关的关键词
        limitation_keywords = [
            "limitation", "drawback", "challenge", "difficulty",
            "restriction", "constraint", "shortcoming", "weakness",
            "need further", "further work", "requires more", "not fully", "lack of",
            "limited to", "cannot", "unable to", "insufficient"
        ]

        # 查找包含这些关键词的句子
        sentences = re.split(r"[.!?]", combined_text)
        for sentence in sentences:
            sentence = sentence.strip()
            if any(keyword in sentence for keyword in limitation_keywords):
                if sentence and len(sentence) < 200:
                    limitations.append(sentence)

        # 没有证据时返回空列表，避免把通用假设伪装成专家意见。
        return limitations[:5]

    @staticmethod
    def _chemistry_context_schema() -> Dict[str, Any]:
        """返回科学上下文的完整固定形状。"""

        nullable_text = {
            "anyOf": [
                {"type": "string", "minLength": 1, "maxLength": 200},
                {"type": "null"},
            ]
        }
        tri_state = {"type": ["boolean", "null"]}
        properties = {
            "reaction_type": {
                "type": "string",
                "enum": [
                    "unknown",
                    "substitution",
                    "addition",
                    "elimination",
                    "catalytic_transformation",
                    "photochemical",
                ],
            },
            "mechanistic_goal": {
                "type": "string",
                "enum": [
                    "unknown",
                    "mechanism_elucidation",
                    "barrier_quantification",
                    "selectivity_rationalization",
                ],
            },
            "suspected_job_types": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["opt", "freq", "ts", "irc", "sp", "excited_state"],
                },
            },
            "solvent": nullable_text,
            "temperature": nullable_text,
            "charge": {
                "anyOf": [
                    {"type": "integer", "minimum": -20, "maximum": 20},
                    {"type": "null"},
                ]
            },
            "multiplicity": {
                "anyOf": [
                    {"type": "integer", "minimum": 1, "maximum": 20},
                    {"type": "null"},
                ]
            },
            "species_roles": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "reactant",
                        "intermediate",
                        "transition state",
                        "product",
                        "catalyst",
                        "conformer",
                    ],
                },
            },
            "candidate_elements": {
                "type": "array",
                "items": {"type": "string", "enum": sorted(_ELEMENT_SYMBOLS)},
            },
            "needs_ts": tri_state,
            "needs_irc": tri_state,
            "needs_excited_state": tri_state,
            "evidence_gaps": {
                "type": "array",
                "items": {"type": "string", "minLength": 1, "maxLength": 300},
            },
        }
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "required": list(properties),
            "properties": properties,
        }

    @staticmethod
    def _validate_chemistry_context(value: Any) -> Dict[str, Any]:
        """规范等价集合，并复核三态需求与任务类型没有互相矛盾。"""

        context = dict(value)
        job_types = set(context["suspected_job_types"])
        pairs = (
            ("needs_ts", "ts"),
            ("needs_irc", "irc"),
            ("needs_excited_state", "excited_state"),
        )
        for flag_name, job_type in pairs:
            flag = context[flag_name]
            # 显式需求已经完整表达了同一科学事实；补入冗余任务枚举不会改变语义。
            if flag is True:
                job_types.add(job_type)
            if flag is False and job_type in job_types:
                raise ValueError(f"{flag_name} 为 false 时不能包含对应任务类型")
        # 数组次序不表达科学含义，冻结前统一排序，避免同一事实只因模型排列不同而
        # 在 A/B 任务中产生无意义差异；重复项同样不携带额外科学信息。
        context["suspected_job_types"] = sorted(job_types)
        for field in ("species_roles", "candidate_elements", "evidence_gaps"):
            context[field] = sorted(set(context[field]))
        return context

    @classmethod
    def validate_frozen_chemistry_context(
        cls,
        value: Mapping[str, Any],
        error: Optional[Mapping[str, Any]],
    ) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """复验恢复前缀只能是完整成功对象，或带原因的空失败对象。"""

        if not isinstance(value, Mapping):
            raise ValueError("冻结科学上下文必须是 object")
        context = dict(value)
        if not context:
            if not isinstance(error, Mapping) or set(error) != {
                "stage",
                "failure_kind",
                "summary",
            }:
                raise ValueError("空科学上下文必须携带完整提取失败说明")
            normalized_error = dict(error)
            if (
                normalized_error.get("stage") != "chemistry_context"
                or normalized_error.get("failure_kind") not in FAILURE_KINDS
                or not isinstance(normalized_error.get("summary"), str)
                or not normalized_error["summary"].strip()
            ):
                raise ValueError("科学上下文提取失败说明非法")
            return {}, normalized_error
        if error is not None:
            raise ValueError("成功科学上下文不能同时携带提取失败说明")
        schema = cls._chemistry_context_schema()
        if not Draft202012Validator(schema).is_valid(context):
            raise ValueError("冻结科学上下文不符合完整固定形状")
        normalized = cls._validate_chemistry_context(context)
        if normalized != context:
            raise ValueError("冻结科学上下文未使用规范数组顺序")
        return normalized, None

    def extract_chemistry_context(
        self,
        question: str,
        *,
        receipt_sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """只依据原题提取一次科学上下文，并对不合格回答进行完整重答。"""

        if not isinstance(question, str) or not question.strip():
            raise RetrievalRootError("chemistry_context", "domain")
        return self._call_retrieval_catalog_stage(
            stage="chemistry_context",
            base_input={"scientific_question": question},
            specification=ModelStage(
                stage="chemistry_context",
                stage_kind="model",
                validation_kind="json_schema",
                schema=self._chemistry_context_schema(),
                domain_validator=self._validate_chemistry_context,
                domain_validator_id="retrieval.chemistry_context.v1",
            ),
            receipt_sink=receipt_sink,
            protocol_regeneration=True,
        )

    def _resolve_chemistry_context(
        self,
        question: str,
        *,
        frozen_context: Optional[Mapping[str, Any]],
        frozen_error: Optional[Mapping[str, Any]],
        supplementary_scientific_input: Optional[Mapping[str, Any]] = None,
        receipt_sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """提取科学上下文，或复验恢复任务已经冻结的提取结果。"""

        if frozen_context is not None:
            # 首次 Retrieval 后半段失败时，恢复任务会带回已经冻结的提取结果。
            # 空对象同样是一次真实的提取失败结果，不能因为它为空就再次调用模型。
            frozen_retrieval_context = dict(frozen_context)
            frozen_supplementary_context = project_supplementary_scientific_input_from_context(
                frozen_retrieval_context
            )
            frozen_retrieval_context.pop("calculation_unit", None)
            frozen_retrieval_context.pop("initial_transition_state_structures", None)
            frozen, frozen_error_value = self.validate_frozen_chemistry_context(
                frozen_retrieval_context,
                frozen_error,
            )
            return merge_supplementary_chemistry_context(
                frozen,
                supplementary_scientific_input or frozen_supplementary_context,
            ), frozen_error_value

        # 科学上下文只从原题抽取一次。它失败时保留真实原因，但检索和后续闭环
        # 仍可直接依赖原题继续，不能把辅助索引失败误当成整个科学任务失败。
        try:
            context = self.extract_chemistry_context(
                question=question,
                receipt_sink=receipt_sink,
            )
        except RetrievalRootError as exc:
            error = {
                "stage": "chemistry_context",
                "failure_kind": exc.failure_kind,
                "summary": "科学上下文提取在允许的回答次数内未通过校验",
            }
            logger.warning(
                "科学上下文提取失败，将冻结空对象并由下游直接读取原题：%s",
                exc.failure_kind,
            )
            return merge_supplementary_chemistry_context(
                {}, supplementary_scientific_input
            ), error
        return merge_supplementary_chemistry_context(
            context,
            supplementary_scientific_input,
        ), None

    # ==================== 主工作流程 ====================

    def process_question(
        self,
        question: str,
        pdf_dir: str = "papers",
        index_dir: str = "index",
        search_papers: bool = True,
        paper_search: Optional[Mapping[str, Any]] = None,
        *,
        frozen_chemistry_context: Optional[Mapping[str, Any]] = None,
        frozen_chemistry_context_error: Optional[Mapping[str, Any]] = None,
        supplementary_scientific_input: Optional[Mapping[str, Any]] = None,
        receipt_sink: Optional[
            Callable[[Mapping[str, Any]], None]
        ] = None,
    ) -> Dict[str, Any]:
        """运行科学上下文、正式 PaperSearch、本地 BGE 索引和文献问答流程。

        ``paper_search`` 是唯一的搜索配置合同。旧调用只传 ``search_papers`` 时
        仍可工作，但会规范化为默认的受控 Scholar provider；搜索未达到目标时仅
        保存可恢复事实并返回，不会把不完整语料送入假设阶段。
        """

        supplementary_context = normalize_supplementary_scientific_input(
            supplementary_scientific_input
        )
        logger.info("开始处理问题: %s...", question[:100])
        chemistry_context, chemistry_context_error = (
            self._resolve_chemistry_context(
                question,
                frozen_context=frozen_chemistry_context,
                frozen_error=frozen_chemistry_context_error,
                supplementary_scientific_input=supplementary_context,
                receipt_sink=receipt_sink,
            )
        )
        keywords = []
        retrieved_papers = []
        answer = ""
        literature_review = ""
        index_built = False
        caught_error = None
        paper_search_status: Optional[Dict[str, Any]] = None
        search_config = normalize_paper_search_config(
            paper_search,
            enabled=search_papers,
        )
        manifest_path = search_config["manifest_path"] or os.path.join(
            os.path.abspath(pdf_dir), ".paper-search-manifest.json"
        )
        search_config["manifest_path"] = os.path.abspath(manifest_path)
        manifest_path_lower = search_config["manifest_path"].lower()
        if manifest_path_lower.endswith(".pdf"):
            raise ValueError("PaperSearch manifest_path 不能指向 PDF 文件")
        if os.path.isfile(search_config["manifest_path"]):
            try:
                if os.path.realpath(search_config["manifest_path"]) in {
                    os.path.realpath(path)
                    for path in self._collect_valid_pdf_paths(pdf_dir)
                }:
                    raise ValueError("PaperSearch manifest_path 不能覆盖语料 PDF")
            except OSError:
                pass
        search_config["resume_state_path"] = os.path.abspath(
            search_config["resume_state_path"]
            or os.path.join(os.path.abspath(pdf_dir), ".paper-search-resume.json")
        )
        search_config["response_cache_dir"] = os.path.abspath(
            search_config["response_cache_dir"]
            or os.path.join(os.path.abspath(pdf_dir), ".paper-search-cache")
        )

        try:
            if search_config["enabled"]:
                keywords = self.extract_keywords(
                    question,
                    receipt_sink=receipt_sink,
                )
                logger.info("提取关键词: %s", keywords)
                keyword_results: List[Dict[str, Any]] = []
                retrieved_papers = self.search_papers(
                    keywords,
                    pdf_dir,
                    status_sink=keyword_results.append,
                    timeout_seconds=search_config["timeout_seconds"],
                    target_papers=search_config["target_papers"],
                    keyword_delay_seconds=search_config["keyword_delay_seconds"],
                    candidate_page_size=search_config["candidate_page_size"],
                    provider_strategy=search_config["provider_strategy"],
                    ranking_question=question,
                    openalex_per_page=search_config["openalex_per_page"],
                    download_delay_seconds=search_config["download_delay_seconds"],
                    scholar_max_pages=search_config["scholar_max_pages"],
                    scholar_results_per_page=search_config["scholar_results_per_page"],
                    scholar_crossref_max_candidates=search_config["scholar_crossref_max_candidates"],
                    scholar_request_delay_seconds=search_config["scholar_request_delay_seconds"],
                    scholar_request_jitter_seconds=search_config["scholar_request_jitter_seconds"],
                    scholar_initial_delay_seconds=search_config["scholar_initial_delay_seconds"],
                    scholar_initial_jitter_seconds=search_config["scholar_initial_jitter_seconds"],
                    scholar_backoff_base_seconds=search_config["scholar_backoff_base_seconds"],
                    scholar_max_backoff_seconds=search_config["scholar_max_backoff_seconds"],
                    scholar_max_consecutive_rate_limits=search_config["scholar_max_consecutive_rate_limits"],
                    keyword_jitter_seconds=search_config["keyword_jitter_seconds"],
                    resume_state_path=search_config["resume_state_path"],
                    response_cache_dir=search_config["response_cache_dir"],
                )
                manifest = self._write_paper_search_manifest(
                    question=question,
                    keywords=keywords,
                    pdf_dir=pdf_dir,
                    manifest_path=manifest_path,
                    target_papers=search_config["target_papers"],
                    search_config=search_config,
                    keyword_results=keyword_results,
                )
                paper_search_status = {
                    "status": "completed" if manifest["target_reached"] else "recoverable_incomplete",
                    "manifest_path": os.path.abspath(manifest_path),
                    "resume_state_path": search_config["resume_state_path"],
                    "response_cache_dir": search_config["response_cache_dir"],
                    "provider": search_config["provider_strategy"],
                    "target_papers": manifest["target_papers"],
                    "actual_papers": manifest["actual_papers"],
                    "target_reached": manifest["target_reached"],
                    "stop_reason": self._paper_search_audit.get("stop_reason"),
                }
                if not manifest["target_reached"]:
                    result = self._normalize_retrieval_result(
                        question=question,
                        keywords=keywords,
                        retrieved_papers=retrieved_papers,
                        answer="",
                        literature_review="",
                        index_built=False,
                    )
                    result["chemistry_context"] = chemistry_context
                    if chemistry_context_error is not None:
                        result["chemistry_context_error"] = chemistry_context_error
                    result["paper_search_status"] = paper_search_status
                    return result
            else:
                # Memory On/Off 的 no-search 入口直接复用 pdf_dir；不调用关键词模型，
                # 避免离线索引流程因为无关的关键词协议失败而无法启动。
                keywords = []
                manifest_path_value = search_config.get("manifest_path")
                if not manifest_path_value or not os.path.isfile(manifest_path_value):
                    raise ValueError("PaperSearch manifest 不存在")
                # 即使调用方没有显式传 manifest，也必须校验默认位置的冻结语料；
                # no-search 只能复用达到目标且与磁盘 PDF 完全一致的 manifest。
                self._validate_frozen_paper_manifest(
                    manifest_path_value,
                    pdf_dir,
                    target_papers=None,
                )

            index_built = bool(self.load_index(index_dir, pdf_dir=pdf_dir))
            if not index_built:
                self.build_index(pdf_dir, index_dir)
            index_built = bool(self.documents)

            if self.documents:
                answer = self.answer_question(
                    question,
                    receipt_sink=receipt_sink,
                )
                literature_review = self.generate_literature_review(
                    question,
                    receipt_sink=receipt_sink,
                )
            logger.info("检索流程完成")
        except Exception as exc:
            caught_error = str(exc)
            logger.error("处理流程失败: %s", caught_error)
            # 即使失败也返回结构化结果

        result = self._normalize_retrieval_result(
            question=question,
            keywords=keywords,
            retrieved_papers=retrieved_papers,
            answer=answer,
            literature_review=literature_review,
            index_built=index_built,
        )
        result["chemistry_context"] = chemistry_context
        if paper_search_status is not None:
            result["paper_search_status"] = paper_search_status
        if chemistry_context_error is not None:
            result["chemistry_context_error"] = chemistry_context_error
        if caught_error is not None:
            result["error"] = caught_error
        return result
