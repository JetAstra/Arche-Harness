"""MiniChem worker stdout 的稳定响应 framing 合同。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


# 结果行必须以固定前缀开始；这样进程退出时追加的第三方 footer 不会被误读成结果。
WORKER_RESPONSE_PREFIX = "MINICHEM_WORKER_RESPONSE_V1:"


def format_worker_response(value: Mapping[str, Any]) -> str:
    """把 worker 结果编码成单行、带版本标记的 JSON 响应。"""

    if not isinstance(value, Mapping):
        raise TypeError("worker response must be a JSON object")
    return (
        WORKER_RESPONSE_PREFIX
        + json.dumps(dict(value), ensure_ascii=False, default=str)
    )


__all__ = ["WORKER_RESPONSE_PREFIX", "format_worker_response"]
