"""保存恢复校验所需的科学问题指纹与目录兼容记录。

本模块不根据题面、Case profile 或历史答案选择工具路线。Action/Backend 的可见性和
路线选择统一来自冻结 MiniChem catalog；这里保留的 ``GaussianRoutePolicy`` 只是旧
调用链使用的兼容类型，以及恢复时核对 scientific question 是否发生漂移的记录。
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping


# 旧 TaskCreated 仍可能保存这两个标识。它们现在都表示“按冻结 catalog 选择”，
# 不再分别启用或禁用任何 Action/Backend。
STRICT_AUTHOR_NATIVE_POLICY_ID = "strict_author_native"
STRUCTURED_EXTENSIONS_POLICY_ID = "structured_extensions"
CATALOG_DECLARED_POLICY_ID = "catalog_declared"

CASE1_FIXED_REPRODUCTION_PROFILE = "case1_fixed_reproduction"
CASE2_FIXED_REPRODUCTION_PROFILE = "case2_fixed_reproduction"
CASE3_FIXED_REPRODUCTION_PROFILE = "case3_fixed_reproduction"
CASE3_AUTHOR_HYPOTHESIS_ADVISORY_PROFILE = "case3_author_hypothesis_advisory"
SUPPORTED_CASE_PROFILES = frozenset(
    {
        CASE1_FIXED_REPRODUCTION_PROFILE,
        CASE2_FIXED_REPRODUCTION_PROFILE,
        CASE3_FIXED_REPRODUCTION_PROFILE,
        CASE3_AUTHOR_HYPOTHESIS_ADVISORY_PROFILE,
    }
)

# typed 专家 patch 与 Executor 共用的可写上界。字段名使用稳定的点分隔路径；
# 当前目录合同仍会进一步限制具体 Action 能修改哪些字段。
TYPED_GAUSSIAN_MUTABLE_FIELDS = frozenset(
    {
        "method_spec.method",
        "method_spec.basis",
        "method_spec.dispersion",
        "method_spec.solvation_model",
        "method_spec.solvent",
        "method_spec.charge",
        "method_spec.multiplicity",
        "action_settings.scf_convergence",
        "action_settings.optimization_convergence",
        "action_settings.max_steps",
    }
)
_POLICY_IDS = frozenset(
    {
        STRICT_AUTHOR_NATIVE_POLICY_ID,
        STRUCTURED_EXTENSIONS_POLICY_ID,
        CATALOG_DECLARED_POLICY_ID,
    }
)
_WHITESPACE = re.compile(r"\s+", re.UNICODE)


class GaussianRoutePolicyError(ValueError):
    """scientific question 或冻结兼容记录不符合恢复校验合同。"""


def is_gaussian_native_step(step: Mapping[str, Any]) -> bool:
    """按 open-execution 的正式软件身份识别 Gaussian native 步骤。

    输入是 canonical workflow step；动作只读取执行层和 ``request.software_id``，
    不读取题面、描述、可执行文件关键词或结果，也不产生副作用。catalog 中的
    software identity 是专家和执行器判断 native Gaussian 的唯一事实来源。
    """

    if not isinstance(step, Mapping) or step.get("execution_layer") != "native_software":
        return False
    request = step.get("request")
    if not isinstance(request, Mapping):
        return False
    software_id = request.get("software_id")
    if not isinstance(software_id, str):
        return False
    return software_id.strip().casefold() == "gaussian"


def _catalog_identity_declares_gaussian(
    action: Mapping[str, Any],
    backend: Mapping[str, Any] | None,
    action_id: str,
    backend_id: str,
) -> bool:
    """从冻结目录的身份元数据判断 pair 是否为 Gaussian 计算能力。

    这里只读取 catalog 明确声明的 software/backend/domain/capability 元数据；不
    根据 scientific question、profile 或固定 Action 名称集合做路由判断。
    """

    del action_id
    if backend_id.strip().casefold() == "gaussian":
        return True

    # 目录可以把软件身份放在 Action 或 Backend 的稳定身份字段中。这里只接受
    # 完整身份值，不扫描描述、能力文本或任意嵌套布尔值，避免相似名称触发路由。
    identity_fields = (
        "software_id",
        "software_family",
        "provider_id",
        "runtime",
        "family",
    )
    accepted_identities = frozenset({"gaussian", "gaussian16", "gaussian_16"})
    for source in (action, backend):
        if not isinstance(source, Mapping):
            continue
        if source.get("gaussian") is True:
            return True
        for field in identity_fields:
            value = source.get(field)
            if (
                isinstance(value, str)
                and value.strip().casefold() in accepted_identities
            ):
                return True
    return False


def is_allowed_typed_gaussian_step(
    step: Mapping[str, Any],
    catalog_snapshot: Mapping[str, Any],
    gaussian_route_policy: "GaussianRoutePolicy | None" = None,
) -> bool:
    """判断步骤是否是目录声明的 typed Gaussian Action/Backend pair。

    ``gaussian_route_policy`` 仅为旧调用链保留，永远不参与 pair 过滤。返回 True
    的条件是：步骤为 predefined Action、Action 和 Backend 身份均存在于同一冻结
    catalog，且目录身份元数据把该 pair 声明为 Gaussian。这样任意 scientific
    question 都能看到相同的 catalog pair。
    """

    del gaussian_route_policy
    if not isinstance(step, Mapping) or not isinstance(catalog_snapshot, Mapping):
        return False
    if step.get("execution_layer", "predefined_actions") != "predefined_actions":
        return False
    action_id = step.get("action_id")
    backend_id = step.get("backend_id")
    if not isinstance(action_id, str) or not action_id.strip():
        return False
    if not isinstance(backend_id, str) or not backend_id.strip():
        return False
    actions = catalog_snapshot.get("actions")
    if not isinstance(actions, list):
        return False
    action: Mapping[str, Any] | None = None
    for candidate in actions:
        if not isinstance(candidate, Mapping):
            continue
        if candidate.get("action_id", candidate.get("id")) == action_id:
            action = candidate
            break
    if action is None:
        return False
    backend_ids = action.get("backend_ids", action.get("provider_ids"))
    if not isinstance(backend_ids, list) or backend_id not in backend_ids:
        return False
    backend: Mapping[str, Any] | None = None
    for candidate in catalog_snapshot.get("backends", []):
        if not isinstance(candidate, Mapping):
            continue
        candidate_id = candidate.get("backend_id", candidate.get("id"))
        if candidate_id == backend_id:
            backend = candidate
            break
    return _catalog_identity_declares_gaussian(action, backend, action_id, backend_id)


def gaussian_step_digest(step: Mapping[str, Any]) -> str:
    """返回 workflow step 的稳定摘要，供审查记录和执行前复验使用。"""

    if not isinstance(step, Mapping):
        raise GaussianRoutePolicyError("Gaussian review step 必须是 object")
    try:
        encoded = json.dumps(
            step,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GaussianRoutePolicyError("Gaussian review step 不是严格 JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def normalize_scientific_question(scientific_question: Any) -> str:
    """规范化题面，供恢复时计算指纹；不参与工具路线选择。"""

    if not isinstance(scientific_question, str):
        raise GaussianRoutePolicyError("scientific_question 必须是字符串")
    normalized = _WHITESPACE.sub(
        " ", unicodedata.normalize("NFKC", scientific_question)
    ).strip()
    if not normalized:
        raise GaussianRoutePolicyError("scientific_question 必须是非空字符串")
    return normalized


def scientific_question_fingerprint(scientific_question: Any) -> str:
    """返回规范化 scientific question 的 SHA-256 指纹。"""

    normalized = normalize_scientific_question(scientific_question)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GaussianRoutePolicy:
    """保存兼容策略标识和题面指纹，不根据它们过滤任何目录 pair。"""

    policy_id: str
    question_fingerprint: str
    case_profile: str | None = None

    def __post_init__(self) -> None:
        """校验持久化身份格式；旧策略标识按同一 catalog 行为处理。"""

        if self.policy_id not in _POLICY_IDS:
            raise GaussianRoutePolicyError("未知 Gaussian route policy id")
        if not isinstance(self.question_fingerprint, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.question_fingerprint
        ):
            raise GaussianRoutePolicyError("question_fingerprint 必须是 64 位小写 SHA-256")
        if self.case_profile is not None and (
            not isinstance(self.case_profile, str) or not self.case_profile.strip()
        ):
            raise GaussianRoutePolicyError("case_profile 必须是非空字符串或 None")

    @property
    def allows_structured_extensions(self) -> bool:
        """兼容旧属性；冻结 catalog 的合法 pair 均可见。"""

        return True

    def allows_typed_gaussian_action(self, action_id: Any, backend_id: Any) -> bool:
        """兼容旧查询；是否可用由 catalog pair 合同决定。"""

        return isinstance(action_id, str) and bool(action_id.strip()) and isinstance(
            backend_id, str
        ) and bool(backend_id.strip())

    def requires_author_native_gaussian_chain_for_pair(
        self,
        action_id: Any,
        backend_id: Any,
    ) -> bool:
        """兼容旧查询；不再强制 codegen/converter/native 三段链。"""

        del action_id, backend_id
        return False

    def to_model_capability_card(self) -> dict[str, Any]:
        """返回不含题面或固定 Action 的通用目录选择事实。"""

        return {"selection_source": "frozen_catalog", "route_selection": "catalog_contract"}

    def to_persisted_record(self) -> dict[str, str]:
        """返回可放入 TaskCreated 的最小稳定恢复记录。"""

        return {
            "policy_id": self.policy_id,
            "question_fingerprint": self.question_fingerprint,
        }


def derive_gaussian_route_policy(
    scientific_question: Any,
    *,
    case_profile: str | None = None,
) -> GaussianRoutePolicy:
    """创建 task-agnostic 兼容记录；所有题面得到相同 catalog 路由行为。"""

    fingerprint = scientific_question_fingerprint(scientific_question)
    return GaussianRoutePolicy(
        policy_id=CATALOG_DECLARED_POLICY_ID,
        question_fingerprint=fingerprint,
        case_profile=case_profile,
    )


def verify_frozen_gaussian_route_policy(
    scientific_question: Any,
    frozen_policy: Any,
    *,
    case_profile: str | None = None,
) -> GaussianRoutePolicy:
    """恢复时核对原题指纹；旧 policy id 只作兼容，不改变当前目录行为。"""

    derived = derive_gaussian_route_policy(
        scientific_question,
        case_profile=case_profile,
    )
    if frozen_policy is None:
        return derived
    if isinstance(frozen_policy, str):
        frozen_id = frozen_policy
        frozen_fingerprint = None
    elif isinstance(frozen_policy, Mapping):
        frozen_id = frozen_policy.get("policy_id")
        frozen_fingerprint = frozen_policy.get("question_fingerprint")
    else:
        raise GaussianRoutePolicyError("冻结 Gaussian route policy 必须是 policy id 或 object")
    if frozen_id not in _POLICY_IDS:
        raise GaussianRoutePolicyError("冻结 Gaussian route policy id 未注册")
    if frozen_fingerprint is not None and frozen_fingerprint != derived.question_fingerprint:
        raise GaussianRoutePolicyError(
            "恢复任务的 scientific question fingerprint 与 TaskCreated 不一致"
        )
    return derived


def validate_persisted_gaussian_route_policy(value: Any) -> dict[str, str]:
    """校验 TaskCreated 保存的最小 route policy 记录。"""

    if not isinstance(value, Mapping) or set(value) != {
        "policy_id",
        "question_fingerprint",
    }:
        raise GaussianRoutePolicyError(
            "持久化 Gaussian route policy 必须恰含 policy_id/question_fingerprint"
        )
    policy = GaussianRoutePolicy(
        policy_id=value["policy_id"],
        question_fingerprint=value["question_fingerprint"],
    )
    return policy.to_persisted_record()


# 便于调用方按“解析/确定”语义使用同一个小接口。
resolve_gaussian_route_policy = derive_gaussian_route_policy


__all__ = [
    "CASE1_FIXED_REPRODUCTION_PROFILE",
    "CASE2_FIXED_REPRODUCTION_PROFILE",
    "CASE3_FIXED_REPRODUCTION_PROFILE",
    "CASE3_AUTHOR_HYPOTHESIS_ADVISORY_PROFILE",
    "CATALOG_DECLARED_POLICY_ID",
    "GaussianRoutePolicy",
    "GaussianRoutePolicyError",
    "STRICT_AUTHOR_NATIVE_POLICY_ID",
    "STRUCTURED_EXTENSIONS_POLICY_ID",
    "SUPPORTED_CASE_PROFILES",
    "TYPED_GAUSSIAN_MUTABLE_FIELDS",
    "derive_gaussian_route_policy",
    "gaussian_step_digest",
    "is_gaussian_native_step",
    "is_allowed_typed_gaussian_step",
    "normalize_scientific_question",
    "resolve_gaussian_route_policy",
    "scientific_question_fingerprint",
    "validate_persisted_gaussian_route_policy",
    "verify_frozen_gaussian_route_policy",
]
