#!/usr/bin/env python3
"""核对 Catalog 与 Investigation MCP 服务是否就绪。

输入是两个 loopback 端口、本次恢复尝试的临时 runtime 根和原任务 runtime 的
Investigation scope 注册表。脚本执行真实 MCP initialize、工具面、catalog 身份
和临时 capability 探针；失败时返回非零状态且不打印 token、配置值或 provider
响应正文。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import secrets
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from minichem_mcp_tools.profiles import (
    CATALOG_BOOTSTRAP_TOOL_NAMES,
    INVESTIGATION_TOOL_NAMES,
)
from minichem_mcp_tools.workspace import ensure_private_runtime_directory
from minichem_toolbox.scope import CAPABILITY_HEADER, ScopeCapabilityRegistry


CATALOG_ENDPOINT = "http://127.0.0.1:9011/mcp"
INVESTIGATION_ENDPOINT = "http://127.0.0.1:9012/mcp"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def validate_fixed_mcp_endpoint(endpoint: str, expected_port: int) -> None:
    """只接受固定 loopback HTTP MCP 地址。"""

    parsed = urlsplit(endpoint)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("MCP endpoint port is invalid") from exc
    if (
        parsed.scheme != "http"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.hostname != "127.0.0.1"
        or parsed.path != "/mcp"
        or parsed.query
        or parsed.fragment
        or port != expected_port
    ):
        raise ValueError("MCP endpoint boundary mismatch")


def decode_text_object(items: object) -> dict[str, object]:
    """从 MCP resource/tool 文本内容解析 JSON object。"""

    for item in items if isinstance(items, list) else ():
        text = getattr(item, "text", None)
        if isinstance(text, str):
            value = json.loads(text)
            if isinstance(value, dict):
                return value
    raise ValueError("MCP payload is not a JSON object")


def decode_tool_object(result: object) -> dict[str, object]:
    """解析 MCP tool 返回值，并拒绝工具错误。"""

    if bool(getattr(result, "isError", False)):
        raise ValueError("MCP tool returned an error")
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured
    return decode_text_object(getattr(result, "content", None))


def read_catalog_identity(payload: Mapping[str, object]) -> tuple[str, str]:
    """核对 9011 公开 catalog 的脱敏身份。"""

    catalog_hash = payload.get("catalog_hash")
    projection_hash = payload.get("catalog_projection_hash")
    schema_version = payload.get("schema_version")
    if (
        not isinstance(catalog_hash, str)
        or not SHA256_PATTERN.fullmatch(catalog_hash)
        or not isinstance(projection_hash, str)
        or not SHA256_PATTERN.fullmatch(projection_hash)
        or isinstance(schema_version, bool)
        or str(schema_version) != "7"
        or payload.get("discovery_mode") != "progressive"
        or payload.get("catalog_visibility_policy")
        != "public_contract_projection"
    ):
        raise ValueError("catalog identity mismatch")
    return catalog_hash, str(schema_version)


def read_investigation_identity(payload: Mapping[str, object]) -> tuple[str, str]:
    """核对 9012 Investigation 的 canonical catalog 身份。"""

    catalog_hash = payload.get("catalog_hash")
    schema_version = payload.get("catalog_version")
    if (
        payload.get("status") != "success"
        or payload.get("catalog_visibility_policy") != "complete_catalog"
        or not isinstance(catalog_hash, str)
        or not SHA256_PATTERN.fullmatch(catalog_hash)
        or isinstance(schema_version, bool)
        or str(schema_version) != "7"
    ):
        raise ValueError("investigation identity mismatch")
    return catalog_hash, str(schema_version)


async def probe_mcp_endpoint(
    endpoint: str,
    expected_tools: tuple[str, ...],
    capability_token: str | None,
) -> tuple[str, str]:
    """对单个 MCP 端点执行 initialize、工具面和身份探针。"""

    headers = (
        {CAPABILITY_HEADER: capability_token}
        if capability_token is not None
        else {}
    )
    timeout = httpx.Timeout(5.0, connect=1.0)
    async with httpx.AsyncClient(
        headers=headers,
        timeout=timeout,
        follow_redirects=False,
    ) as client:
        async with streamable_http_client(endpoint, http_client=client) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                initialized = await session.initialize()
                if not getattr(initialized, "protocolVersion", None) or not getattr(
                    getattr(initialized, "serverInfo", None), "name", None
                ):
                    raise ValueError("MCP server identity is missing")
                tools = await session.list_tools()
                if tuple(tool.name for tool in tools.tools) != expected_tools:
                    raise ValueError("MCP tool surface mismatch")
                if capability_token is None:
                    resource = await session.read_resource("minichem://catalog")
                    return read_catalog_identity(
                        decode_text_object(getattr(resource, "contents", None))
                    )
                result = await session.call_tool(
                    "list_action_domains",
                    {"request": {"include_action_ids": False}},
                )
                return read_investigation_identity(decode_tool_object(result))


async def probe_mcp_service_pair(
    capability_token: str,
    catalog_endpoint: str,
    investigation_endpoint: str,
) -> None:
    """核对两端点并确认它们冻结同一 catalog。"""

    catalog_identity = await probe_mcp_endpoint(
        catalog_endpoint,
        CATALOG_BOOTSTRAP_TOOL_NAMES,
        None,
    )
    investigation_identity = await probe_mcp_endpoint(
        investigation_endpoint,
        INVESTIGATION_TOOL_NAMES,
        capability_token,
    )
    if catalog_identity != investigation_identity:
        raise ValueError("MCP services do not share catalog identity")


def parse_readiness_arguments() -> argparse.Namespace:
    """解析恢复入口传入的临时目录和原 scope 注册表。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--scope-registry", required=True)
    parser.add_argument("--catalog-port", type=int, default=9011)
    parser.add_argument("--investigation-port", type=int, default=9012)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    for option_name in ("catalog_port", "investigation_port"):
        port = getattr(args, option_name)
        if not 1 <= port <= 65535:
            parser.error(f"--{option_name.replace('_', '-')} 必须位于 1 到 65535 之间")
    if args.catalog_port == args.investigation_port:
        parser.error("--catalog-port 与 --investigation-port 不能相同")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds 必须大于 0")
    return args


def run_readiness_probe() -> int:
    """在超时前重复真实 MCP readiness，成功后撤销临时 capability。"""

    args = parse_readiness_arguments()
    catalog_endpoint = f"http://127.0.0.1:{args.catalog_port}/mcp"
    investigation_endpoint = f"http://127.0.0.1:{args.investigation_port}/mcp"
    validate_fixed_mcp_endpoint(catalog_endpoint, args.catalog_port)
    validate_fixed_mcp_endpoint(investigation_endpoint, args.investigation_port)

    runtime_root = Path(args.runtime_root).resolve(strict=False)
    registry_root = Path(args.scope_registry).resolve(strict=False)
    readiness_research_task_id = f"readiness_{secrets.token_hex(8)}"
    readiness_workspace = runtime_root / readiness_research_task_id
    ensure_private_runtime_directory(
        str(readiness_workspace),
        runtime_root=str(runtime_root),
    )
    registry = ScopeCapabilityRegistry(str(registry_root))
    # ScopeCapabilityRegistry 是第三方服务协议；其 task_id 字段在此边界
    # 映射本次临时 research_task 身份，ARCHE 上层不暴露 task_id。
    capability = registry.provision_scope_capability(
        task_id=readiness_research_task_id,
        batch_id=readiness_research_task_id,
        combination_id=readiness_research_task_id,
        scope_id=readiness_research_task_id,
        workspace=str(readiness_workspace),
        token=secrets.token_urlsafe(32),
    )
    deadline = time.monotonic() + args.timeout_seconds
    try:
        while True:
            try:
                asyncio.run(
                    probe_mcp_service_pair(
                        capability.token,
                        catalog_endpoint,
                        investigation_endpoint,
                    )
                )
                print("MCP readiness passed")
                return 0
            except Exception:
                if time.monotonic() >= deadline:
                    raise SystemExit("MCP readiness timed out")
                time.sleep(1.0)
    finally:
        registry.revoke_scope_capability(capability.token)


if __name__ == "__main__":
    raise SystemExit(run_readiness_probe())
