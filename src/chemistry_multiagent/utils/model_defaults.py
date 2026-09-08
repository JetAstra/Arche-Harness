"""集中定义通用模型和专家回退模型的默认标识。

通用 Retrieval、Hypothesis、Planner、Router、Revision 和专家角色都由同一个
``ARCHE_CHEM_MODEL_NAME`` 控制。未配置时保留稳定的 Pro 默认值；假设形成的 Agent A1
只有在显式切换 provider 且三项独立配置完整时才启用。
"""

from __future__ import annotations

import os


DEFAULT_GENERAL_MODEL = "deepseek-v4-pro"
DEFAULT_EXPERT_FALLBACK_MODEL = DEFAULT_GENERAL_MODEL
DEFAULT_CODEX_MODEL = "gpt-5.6-sol"
LEGACY_GENERAL_MODEL_NAMES = frozenset({"deepseek-chat", "deepseek-reasoner"})
# Sol provider 使用独立的三元组配置，不能借用 ARCHE_CHEM 的密钥或地址。
CODEX_API_KEY_ENV = "ARCHE_CODEX_API_KEY"
CODEX_BASE_URL_ENV = "ARCHE_CODEX_BASE_URL"
CODEX_MODEL_NAME_ENV = "ARCHE_CODEX_MODEL_NAME"
CODEX_REASONING_EFFORT = "xhigh"
# 假设形成六个阶段的 provider 选择开关；未配置时保留既有 Codex 路由。
HYPOTHESIS_FORMATION_PROVIDER_ENV = "ARCHE_HYPOTHESIS_FORMATION_PROVIDER"
HYPOTHESIS_FORMATION_PROVIDER_CODEX = "codex"
HYPOTHESIS_FORMATION_PROVIDER_AGENT_A1 = "agent_a1"
HYPOTHESIS_FORMATION_PROVIDER_CLUSTER_ALIAS = "cluster"
# Agent A1 是独立的 OpenAI-compatible Chat Completions provider，使用 Bearer API key。
AGENT_A1_BASE_URL_ENV = "ARCHE_AGENT_A1_BASE_URL"
AGENT_A1_API_KEY_ENV = "ARCHE_AGENT_A1_API_KEY"
AGENT_A1_MODEL_NAME_ENV = "ARCHE_AGENT_A1_MODEL_NAME"
AGENT_A1_MODEL_NAME = "Agents-A1"
# 只写入 compiled request 和 trace 的公开 provider 身份，不包含地址或认证信息。
CODEX_RESPONSES_PROVIDER = "codex_responses"
AGENT_A1_CHAT_COMPLETIONS_PROVIDER = "agent_a1_chat_completions"
ARCHE_CHEM_CHAT_COMPLETIONS_PROVIDER = "arche_chem_chat_completions"
# 未设置环境变量时沿用此前的最高推理强度。
DEEPSEEK_V4_PRO_REASONING_EFFORT = "max"
DEEPSEEK_V4_PRO_REASONING_EFFORT_ENV = "DEEPSEEK_V4_PRO_REASONING_EFFORT"
DEEPSEEK_V4_PRO_REASONING_EFFORT_VALUES = frozenset({"max", "high"})


def resolve_deepseek_v4_pro_reasoning_effort() -> str:
    """解析 DeepSeek v4 Pro 的推理强度并拒绝未登记配置。

    输入：进程环境中的 ``DEEPSEEK_V4_PRO_REASONING_EFFORT``。动作：读取并清理
    环境值；未配置时返回兼容旧行为的 ``max``，否则只接受 ``max`` 或 ``high``。
    返回：本次 Pro 请求应发送且应写入审计的实际值。失败：环境值为空白或不在
    登记集合中时，在任何网络请求前抛出 ``ValueError``。
    """

    configured = os.environ.get(DEEPSEEK_V4_PRO_REASONING_EFFORT_ENV)
    if configured is None:
        return DEEPSEEK_V4_PRO_REASONING_EFFORT
    resolved = configured.strip()
    if resolved not in DEEPSEEK_V4_PRO_REASONING_EFFORT_VALUES:
        allowed = ", ".join(sorted(DEEPSEEK_V4_PRO_REASONING_EFFORT_VALUES))
        raise ValueError(
            f"{DEEPSEEK_V4_PRO_REASONING_EFFORT_ENV} 必须是 {allowed} 之一"
        )
    return resolved


def resolve_general_model(model: str | None = None) -> str:
    """解析通用模型名称并把历史别名映射到实际模型。

    输入：可选的显式模型名称以及进程中的 ``ARCHE_CHEM_MODEL_NAME``。动作：显式模型
    优先；未提供显式模型时读取统一环境变量；环境变量为空或历史
    ``deepseek-chat``/``deepseek-reasoner`` 别名回到 Pro 默认值。返回：可直接
    发送给通用 API 的非空模型名称。失败：传入非字符串或空白显式名称时抛出
    ``TypeError``。
    """

    configured = os.environ.get("ARCHE_CHEM_MODEL_NAME", "").strip()
    if not configured or configured in LEGACY_GENERAL_MODEL_NAMES:
        configured = DEFAULT_GENERAL_MODEL

    if model is None:
        return configured
    if not isinstance(model, str):
        raise TypeError("model 必须是字符串或 None")
    requested = model.strip()
    if not requested:
        raise TypeError("model 必须是非空字符串或 None")
    if requested in LEGACY_GENERAL_MODEL_NAMES:
        return configured
    return requested


def resolve_codex_provider_settings() -> tuple[str, str, str]:
    """读取 Sol Responses provider 的独立配置并校验三项均存在。

    输入：进程环境中的 ``ARCHE_CODEX_MODEL_NAME``、``ARCHE_CODEX_BASE_URL``
    和 ``ARCHE_CODEX_API_KEY``。动作：读取并去除首尾空白，只检查三项是否存在；
    返回：按 ``(model, base_url, api_key)`` 排列的配置。失败：任一项为空时在
    网络请求前抛出只包含变量名的 ``ValueError``，不输出密钥或地址正文，也不会
    回退到 ARCHE_CHEM 配置。
    """

    values = {
        CODEX_MODEL_NAME_ENV: os.environ.get(CODEX_MODEL_NAME_ENV, "").strip(),
        CODEX_BASE_URL_ENV: os.environ.get(CODEX_BASE_URL_ENV, "").strip(),
        CODEX_API_KEY_ENV: os.environ.get(CODEX_API_KEY_ENV, "").strip(),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ValueError(
            "Codex provider 缺少独立配置；缺少：" + ", ".join(missing)
        )
    return (
        values[CODEX_MODEL_NAME_ENV],
        values[CODEX_BASE_URL_ENV],
        values[CODEX_API_KEY_ENV],
    )


def resolve_hypothesis_formation_provider() -> str:
    """解析假设形成 provider 开关并拒绝未登记值。

    输入：进程环境中的 ``ARCHE_HYPOTHESIS_FORMATION_PROVIDER``。动作：读取并
    清理开关，将兼容别名 ``cluster`` 归一为 ``agent_a1``。返回：``codex`` 或
    ``agent_a1``。失败：环境变量显式为空或不是登记值时，在请求编译前抛出
    ``ValueError``，不会静默切换模型。
    """

    configured = os.environ.get(HYPOTHESIS_FORMATION_PROVIDER_ENV)
    if configured is None:
        return HYPOTHESIS_FORMATION_PROVIDER_CODEX
    normalized = configured.strip().casefold()
    if normalized == HYPOTHESIS_FORMATION_PROVIDER_CLUSTER_ALIAS:
        return HYPOTHESIS_FORMATION_PROVIDER_AGENT_A1
    if normalized in {
        HYPOTHESIS_FORMATION_PROVIDER_CODEX,
        HYPOTHESIS_FORMATION_PROVIDER_AGENT_A1,
    }:
        return normalized
    raise ValueError(
        f"{HYPOTHESIS_FORMATION_PROVIDER_ENV} 必须是 codex、agent_a1 或 cluster"
    )


def resolve_agent_a1_provider_settings() -> tuple[str, str, str]:
    """读取独立 Agent A1 配置并在缺项时 fail-closed。

    输入：进程环境中的 ``ARCHE_AGENT_A1_BASE_URL``、``ARCHE_AGENT_A1_API_KEY``
    和 ``ARCHE_AGENT_A1_MODEL_NAME``。动作：清理三项配置，检查它们均存在，并
    模型名称由该 provider 自己配置；返回：按模型、地址、Bearer API key 排列的
    内部配置。失败：任一项缺失时，只报告变量名，不输出配置正文，也不创建网络
    客户端。模型名称不再绑定某个历史登记字符串。
    """

    values = {
        AGENT_A1_BASE_URL_ENV: os.environ.get(AGENT_A1_BASE_URL_ENV, "").strip(),
        AGENT_A1_API_KEY_ENV: os.environ.get(AGENT_A1_API_KEY_ENV, "").strip(),
        AGENT_A1_MODEL_NAME_ENV: os.environ.get(
            AGENT_A1_MODEL_NAME_ENV, ""
        ).strip(),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ValueError(
            "Agent A1 provider 缺少独立配置；缺少：" + ", ".join(missing)
        )
    return (
        values[AGENT_A1_MODEL_NAME_ENV],
        values[AGENT_A1_BASE_URL_ENV],
        values[AGENT_A1_API_KEY_ENV],
    )


def is_codex_model(model: str | None) -> bool:
    """判断模型标识是否应走独立 Sol Responses provider。

    输入是 compiled request 中的模型标识；动作只比较固定的 Sol 默认名称或当前
    独立环境配置的模型名称，不做模糊匹配。返回布尔值；配置缺失时不会把 Sol
    请求静默改投 ARCHE_CHEM。
    """

    if not isinstance(model, str):
        return False
    normalized = model.strip()
    if not normalized:
        return False
    configured = os.environ.get(CODEX_MODEL_NAME_ENV, "").strip()
    return normalized == DEFAULT_CODEX_MODEL or (
        bool(configured) and normalized == configured
    )


def is_deepseek_v4_pro_model(model: str | None) -> bool:
    """判断模型标识是否明确属于 DeepSeek v4 Pro。

    输入是模型公开标识；动作去除命名空间和首尾空白，并接受官方基础名称及数字
    版本后缀（例如 ``deepseek-v4-pro-0813``）。返回值为布尔值；空值、Flash、
    本地模型和未登记的相似名称返回 ``False``，不会据此发送 Pro 专用参数。
    """

    if not isinstance(model, str):
        return False
    basename = model.strip().rsplit("/", 1)[-1].casefold()
    if basename == DEFAULT_GENERAL_MODEL:
        return True
    prefix = f"{DEFAULT_GENERAL_MODEL}-"
    suffix = basename[len(prefix) :] if basename.startswith(prefix) else ""
    return bool(suffix) and suffix.isdigit()


__all__ = [
    "CODEX_API_KEY_ENV",
    "CODEX_BASE_URL_ENV",
    "CODEX_MODEL_NAME_ENV",
    "CODEX_REASONING_EFFORT",
    "HYPOTHESIS_FORMATION_PROVIDER_ENV",
    "HYPOTHESIS_FORMATION_PROVIDER_CODEX",
    "HYPOTHESIS_FORMATION_PROVIDER_AGENT_A1",
    "HYPOTHESIS_FORMATION_PROVIDER_CLUSTER_ALIAS",
    "AGENT_A1_BASE_URL_ENV",
    "AGENT_A1_API_KEY_ENV",
    "AGENT_A1_MODEL_NAME_ENV",
    "AGENT_A1_MODEL_NAME",
    "CODEX_RESPONSES_PROVIDER",
    "AGENT_A1_CHAT_COMPLETIONS_PROVIDER",
    "ARCHE_CHEM_CHAT_COMPLETIONS_PROVIDER",
    "DEFAULT_CODEX_MODEL",
    "DEFAULT_EXPERT_FALLBACK_MODEL",
    "DEFAULT_GENERAL_MODEL",
    "DEEPSEEK_V4_PRO_REASONING_EFFORT",
    "DEEPSEEK_V4_PRO_REASONING_EFFORT_ENV",
    "DEEPSEEK_V4_PRO_REASONING_EFFORT_VALUES",
    "LEGACY_GENERAL_MODEL_NAMES",
    "is_codex_model",
    "is_deepseek_v4_pro_model",
    "resolve_codex_provider_settings",
    "resolve_hypothesis_formation_provider",
    "resolve_agent_a1_provider_settings",
    "resolve_deepseek_v4_pro_reasoning_effort",
    "resolve_general_model",
]
