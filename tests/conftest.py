"""让默认测试保持离线且结果可重复。

真实命令会主动加载 ``config.local.env``，其中可能配置非 React Planner。
替换旧 OpenAI 客户端的单元测试不得因此启动外部 Harness；需要测试其他
Harness 的用例会显式覆盖本夹具。
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _default_react_harness_for_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARCHE_PLANNER_DRIVER", "react")
    # 本地配置可能选择编码模型。旧 API 合同测试固定使用 DeepSeek Pro；
    # 需要其他模型的用例会单独覆盖此值。
    monkeypatch.setenv("ARCHE_CHEM_MODEL_NAME", "deepseek-v4-pro")
    # MiniChem 目录合同会比较公开资源政策与仓库内的固定政策。本机为低资源
    # smoke 服务设置的覆盖值不得改变单元测试验证的合同。
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES", "60")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB", "320000")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT", "1")
