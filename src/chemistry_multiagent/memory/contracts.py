"""定义 Working Memory 能保存的事实，以及并发写入时如何避免相互覆盖。

所有事件内容、任务状态和路径都会先在这里检查。恢复运行时因此可以沿着一条
完整、可追溯的事实链继续，而不需要猜测数据库里缺失或多余字段的含义。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from pathlib import PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from chemistry_multiagent.utils.model_identity import model_identity_matches
from chemistry_multiagent.utils.gaussian_routing import (
    CASE1_FIXED_REPRODUCTION_PROFILE,
    CASE2_FIXED_REPRODUCTION_PROFILE,
    CASE3_FIXED_REPRODUCTION_PROFILE,
    CASE3_AUTHOR_HYPOTHESIS_ADVISORY_PROFILE,
    SUPPORTED_CASE_PROFILES,
    validate_persisted_gaussian_route_policy,
)


# 任务状态描述整个实验是否仍可调度。等待外部程序和等待人工处理要分开记录，
# 否则恢复器无法判断应该继续轮询 Gaussian，还是应该先展示恢复说明。
TASK_STATES = {
    "running",
    "waiting_execution",
    "paused",
    "failed",
    "succeeded",
    "stopped_no_supported_hypothesis",
}
LEGACY_TERMINAL_TASK_STATES = {"completed", "stopped"}
# 批次只在其中所有假设组合都停止推进后才能结束；组合状态则保留更细的模型、
# 工具和修订阶段，供调度器决定下一次只能调用哪一个组件。
BATCH_STATES = {"running", "succeeded", "failed"}
COMBINATION_STATES = {
    "generating_workflow",
    "running",
    "waiting_execution",
    "routing",
    "revising_workflow",
    "paused",
    "evidence_ready",
    "hypothesis_failed",
    "tool_capability_gap",
}
ACTIVE_COMBINATION_STATES = {
    "generating_workflow",
    "running",
    "routing",
    "revising_workflow",
}
TERMINAL_COMBINATION_STATES = {
    "evidence_ready",
    "hypothesis_failed",
    "tool_capability_gap",
}
# 三种组合终局都能参加批次归约；能力缺口只表示当前工具目录无法覆盖验证需求，
# 不会因此改写为科学失败或建立新的任务级状态。
BATCH_CLOSABLE_COMBINATION_STATES = {
    "evidence_ready",
    "hypothesis_failed",
    "tool_capability_gap",
}
# 技术暂停不是科学终态。只有同批已经得到可信证据时，批次结果才会把尚未解决的
# 暂停记作 technical_pause，既保留真实故障，也不让它覆盖已经成立的成功结果。
BATCH_TECHNICAL_PAUSE_STATE = "technical_pause"


def batch_is_ready_to_publish(states: Iterable[str]) -> bool:
    """判断当前批次是否已经具备一次性发布条件。

    常规情况下，每个组合都必须形成科学终态或工具能力缺口。若至少一个组合已经
    证据就绪，其余组合可以停在技术暂停；仍在执行或等待外部作业的组合不会被提前
    截断。全批只有技术暂停时也继续等待研究员恢复。
    """

    values = list(states)
    if not values:
        return False
    if all(item in BATCH_CLOSABLE_COMBINATION_STATES for item in values):
        return True
    return "evidence_ready" in values and all(
        item in BATCH_CLOSABLE_COMBINATION_STATES or item == "paused"
        for item in values
    )
# 这些事件共同组成可重建的历史主线。检查点和恢复记录外部作业的生命周期，
# Attempt（一次完整科学尝试）记录执行、程序检查与 Router 决定的闭环结果。
PRIMARY_EVENT_TYPES = {
    "task_created",
    "hypothesis_batch_created",
    "planning_completed",
    "tool_capability_gap",
    "execution_step_completed",
    "execution_step_failed",
    "execution_request_rebound",
    "execution_checkpointed",
    "execution_resumed",
    "attempt",
    "execution_paused",
    "restart_applied",
    "hypothesis_batch_closed",
    "task_closed",
    "expert_gaussian_review",
    "expert_debate_completed",
    "expert_model_paused",
    "retrieval_completed",
    "retrieval_paused",
}

# 任务在创建事实中冻结所使用的阶段实现。这个值属于任务事实，不从当前进程的
# 环境变量推断；因此恢复旧任务时即使环境已经切换，也仍然使用原来的阶段。
STAGE_ARCHITECTURE_VERSIONS = frozenset(
    {
        "legacy_workflow_v1",
        "investigation_v2",
    }
)
LEGACY_STAGE_ARCHITECTURE_VERSION = "legacy_workflow_v1"
INVESTIGATION_STAGE_ARCHITECTURE_VERSION = "investigation_v2"
INVESTIGATION_PROTOCOL_VERSION = 8
# 新任务入口在 TaskCreated 中冻结的执行范围。旧任务缺少该字段时按
# ``end_to_end`` 解释，以保持历史 v2 任务的恢复兼容性。
INVESTIGATION_RUN_MODE_FIELD = "run_mode"
INVESTIGATION_RUN_MODES = frozenset({"end_to_end", "top5_selection"})

# Investigation 事件使用独立的事实名称，不能被当前状态计算当作旧 workflow、Attempt
# 或 revision。它们仍然共用 memory_events 的追加和来源引用机制。
INVESTIGATION_EVENT_TYPES = {
    "investigation_session_started",
    "investigation_session_thread_bound",
    "investigation_session_thread_migrated",
    "investigation_turn_started",
    "investigation_turn_waiting",
    "investigation_turn_attempt_terminated",
    "investigation_legacy_waiting_resume_authorized",
    "investigation_incomplete_v2_waiting_resume_authorized",
    "investigation_result_recorded",
    "investigation_batch_reviewed",
    "investigation_repair_requested",
    "investigation_repair_completed",
    "investigation_route_marked",
    "investigation_evidence_verified",
    "task_review_session_started",
    "task_review_turn_started",
    "investigation_repair_turn_started",
    "investigation_task_closed",
    "investigation_task_failed",
    "investigation_task_correction_recorded",
    "hypothesis_selection_completed",
}
PRIMARY_EVENT_TYPES |= INVESTIGATION_EVENT_TYPES
# 一次模型或工具操作只能产生下列一种主要结果。检查点与恢复只是同一工具操作的
# 中间状态，因此不在这里占用操作身份，也不消费人工恢复授权。
OPERATION_TERMINAL_EVENT_TYPES = {
    "hypothesis_batch_created",
    "planning_completed",
    "tool_capability_gap",
    "execution_step_completed",
    "execution_step_failed",
    "attempt",
    "execution_paused",
    "expert_gaussian_review",
    "expert_debate_completed",
    "expert_model_paused",
    "retrieval_completed",
    "retrieval_paused",
}


def validate_retrieval_completed_payload(payload: Any) -> Dict[str, Any]:
    """校验 Retrieval 成功事实及其可复核结果 artifact。"""

    value = require_exact_keys(
        payload,
        {
            "operation_instance_id",
            "operation",
            "status",
            "result_digest",
            "artifact_ref",
            "causal_source_ref",
        },
        "RetrievalCompleted.payload",
    )
    require_nonempty_string(value["operation_instance_id"], "operation_instance_id")
    if value["operation"] != "retrieval":
        raise ContractValidationError("RetrievalCompleted.operation 必须为 retrieval")
    if value["status"] != "completed":
        raise ContractValidationError("RetrievalCompleted.status 必须为 completed")
    if not isinstance(value["result_digest"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", value["result_digest"]
    ):
        raise ContractValidationError("RetrievalCompleted.result_digest 必须是 64 位小写 SHA-256")
    validate_task_relative_path(value["artifact_ref"], "artifact_ref")
    if value["causal_source_ref"] is not None:
        require_nonempty_string(value["causal_source_ref"], "causal_source_ref")
    canonical_json(value)
    return dict(value)


def validate_retrieval_pause_payload(payload: Any) -> Dict[str, Any]:
    """校验 PaperSearch 未完成时保存的任务级暂停事实。

    三个检索路径只保存脱敏后的逻辑 basename。这样数据库和任务 artifact 不会
    暴露运行机器的绝对目录，也不会让恢复器沿着 ``..`` 访问任务目录之外。
    """

    value = require_exact_keys(
        payload,
        {
            "operation_instance_id",
            "operation",
            "reason_code",
            "provider",
            "manifest_path",
            "resume_state_path",
            "response_cache_dir",
            "target_papers",
            "actual_papers",
            "target_reached",
            "stop_reason",
            "artifact_ref",
        },
        "RetrievalPaused.payload",
    )
    if value["operation"] != "paper_search":
        raise ContractValidationError("RetrievalPaused.operation 必须是 paper_search")
    for field in (
        "operation_instance_id",
        "reason_code",
        "provider",
        "manifest_path",
        "resume_state_path",
        "response_cache_dir",
        "artifact_ref",
    ):
        require_nonempty_string(value[field], field)
    for field in ("manifest_path", "resume_state_path", "response_cache_dir"):
        path = value[field]
        posix = PurePosixPath(path)
        if (
            "\\" in path
            or posix.is_absolute()
            or len(posix.parts) != 1
            or posix.parts[0] in {"", ".", ".."}
            or str(posix) != path
        ):
            raise ContractValidationError(
                f"{field} 必须是脱敏后的逻辑 basename，不能是绝对路径或包含 .."
            )
    validate_task_relative_path(value["artifact_ref"], "artifact_ref")
    require_json_integer(value["target_papers"], "target_papers", 1)
    require_json_integer(value["actual_papers"], "actual_papers", 0)
    if not isinstance(value["target_reached"], bool) or value["target_reached"]:
        raise ContractValidationError("RetrievalPaused.target_reached 必须为 false")
    if value["stop_reason"] is not None:
        require_nonempty_string(value["stop_reason"], "stop_reason")
    canonical_json(value)
    return dict(value)
# 成功步骤会结束当前工具调用，但不会消费人工恢复授权。这个授权要沿着后续步骤
# 继续传到 Attempt；只有真正形成科学决定或再次暂停时才算被使用。
# 单步完成或失败只是同一执行尝试的中间事实，后面的 Attempt 才消费人工恢复
# 授权；否则恢复后的第一步结果会阻止同一 execution 提交最终科学判断。
CAUSAL_SOURCE_CONSUMER_EVENT_TYPES = OPERATION_TERMINAL_EVENT_TYPES - {
    "execution_step_completed",
    "execution_step_failed",
}
TASK_CONFIG_FIELDS = {
    "num_queries",
    "num_hypotheses_per_query",
    "top_n_workflows",
    "max_parallel_combinations",
    "max_workflow_revisions_per_combination",
    "max_hypothesis_batches",
}
# Investigation v2 使用独立的任务配置合同。旧 workflow 字段只由
# ``validate_task_config`` 和 legacy_workflow_v1 路径读取，不能在 v2 事实中混入。
INVESTIGATION_TASK_CONFIG_FIELDS = {
    "num_queries",
    "num_hypotheses_per_query",
    "top_n_hypotheses",
    "max_parallel_investigations",
    "max_investigation_repairs_per_combination",
    "max_hypothesis_batches",
}
# 新任务可以把当前批次的组合数量显式冻结在任务配置中。旧 TaskCreated 没有这个
# 可选字段时，读取方按 top_n_hypotheses 推导，因此现有五组合任务的事实格式不变。
INVESTIGATION_COMBINATION_COUNT_FIELD = "investigation_combination_count"
INVESTIGATION_MODE_FIELD = "investigation_mode"
# 作者主假设模式只保留已经登记的 Case 适配器；具体夹具文件由 author
# importer 读取，Working Memory 这里只核对公开身份和单组合预算。
AUTHOR_MAIN_HYPOTHESIS_MODES = frozenset(
    {
        "author_case1_main_hypothesis",
        "author_case3_main_hypothesis",
        "author_case3_advisory_main_hypothesis",
        "author_case3_main_hypothesis_only",
        "author_case3_main_hypothesis_exploration",
    }
)
AUTHOR_MAIN_HYPOTHESIS_FIXTURE_IDS = {
    "author_case1_main_hypothesis": "case1_author_main_hypothesis",
    "author_case3_main_hypothesis": "case3_author_main_hypothesis",
    "author_case3_advisory_main_hypothesis": "case3_author_main_hypothesis",
    "author_case3_main_hypothesis_only": "case3_author_main_hypothesis_only",
    "author_case3_main_hypothesis_exploration": "case3_author_main_hypothesis_exploration",
}
AUTHOR_MAIN_HYPOTHESIS_CASE_PROFILES = {
    "author_case1_main_hypothesis": "case1_fixed_reproduction",
    "author_case3_main_hypothesis": "case3_fixed_reproduction",
    "author_case3_advisory_main_hypothesis": (
        CASE3_AUTHOR_HYPOTHESIS_ADVISORY_PROFILE
    ),
    "author_case3_main_hypothesis_only": CASE3_FIXED_REPRODUCTION_PROFILE,
    "author_case3_main_hypothesis_exploration": CASE3_FIXED_REPRODUCTION_PROFILE,
}
MODEL_OPERATIONS = {
    "workflow_generation",
    "workflow_optimization",
    "router",
    "workflow_revision",
    "expert_gaussian_review",
    "expert_debate",
}
PAUSE_OPERATIONS = MODEL_OPERATIONS | {
    "hypothesis_generation",
    "execute_step",
}
PAUSE_REASON_OPERATION_SCOPE = {
    ("model_protocol_failure", "combination", "workflow_generation"),
    ("model_protocol_failure", "combination", "router"),
    ("model_protocol_failure", "combination", "workflow_revision"),
    ("model_service_failure", "combination", "workflow_generation"),
    ("model_protocol_failure", "combination", "workflow_optimization"),
    ("model_service_failure", "combination", "workflow_optimization"),
    ("model_service_failure", "combination", "router"),
    ("model_service_failure", "combination", "workflow_revision"),
    ("model_protocol_failure", "combination", "expert_gaussian_review"),
    ("model_service_failure", "combination", "expert_gaussian_review"),
    ("unknown_in_flight_request", "combination", "expert_gaussian_review"),
    ("model_protocol_failure", "combination", "expert_debate"),
    ("model_service_failure", "combination", "expert_debate"),
    ("unknown_in_flight_request", "combination", "expert_debate"),
    ("tool_execution_failure", "combination", "execute_step"),
    ("tool_execution_failure", "task", "hypothesis_generation"),
    ("model_protocol_failure", "task", "hypothesis_generation"),
    ("model_service_failure", "task", "hypothesis_generation"),
}
# Controller 的实际状态转换必须与 Router 的科学决定逐项对应。单独维护这张映射，
# 可以在写入历史前发现“模型说修订工作流、程序却结束假设”之类的错接。
ATTEMPT_ACTION_DECISION = {
    "mark_evidence_ready": "evidence_ready",
    "continue_workflow_revision": "revise_workflow",
    "mark_hypothesis_failed": "revise_hypothesis",
}
# 终态原因用于向人解释组合为何结束。人工放弃不属于模型 Attempt，因而只出现在
# 更宽的组合终态集合中，不能伪装成一次已经运行过 Router 的科学尝试。
ATTEMPT_TERMINAL_REASONS = {
    "router_evidence_ready",
    "router_revise_hypothesis",
    "workflow_revision_limit_exhausted",
}
COMBINATION_TERMINAL_REASONS = ATTEMPT_TERMINAL_REASONS | {
    "human_revise_hypothesis",
    "tool_capability_gap",
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA256_IN_NAME_RE = re.compile(r"(?<![0-9a-f])([0-9a-f]{64})(?![0-9a-f])")


class WorkingMemoryError(RuntimeError):
    """Working Memory 拒绝输入或写入时使用的公共异常类型。"""


class ContractValidationError(WorkingMemoryError, ValueError):
    """待保存内容缺字段、类型错误或前后关系不成立。"""


def validate_case_profile(value: Any, *, allow_none: bool = True) -> str | None:
    """校验固定复现标识并返回规范值。

    输入是任务入口、恢复事实或模型提示使用的内部标识；函数只允许当前三种
    固定复现 profile，返回值不会被序列化进模型 user 消息。``None`` 表示普通
    任务，除非调用方显式关闭 ``allow_none``；其他值立即抛出合同错误。
    """

    if value is None and allow_none:
        return None
    if not isinstance(value, str) or value not in SUPPORTED_CASE_PROFILES:
        raise ContractValidationError("case_profile 不是受支持的固定复现标识")
    return value


class IdempotencyConflict(WorkingMemoryError):
    """同一个重复请求标识曾经保存过不同内容，不能安全当作重放。"""


class OperationConflict(WorkingMemoryError):
    """同一次运行已经结束，而当前结果与先前保存的终局结果冲突。"""


class MemoryDisabledError(WorkingMemoryError):
    """无记忆模式误触发数据库写入时立即报错，防止基线受历史数据污染。"""


class CASConflict(WorkingMemoryError):
    """另一个执行者已推进任务，当前写入依据的旧状态不再有效。"""

    def __init__(
        self,
        message: str,
        *,
        expected_version: Optional[int],
        current_version: Optional[int],
        expected_state: Optional[str] = None,
        current_state: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.expected_version = expected_version
        self.current_version = current_version
        self.expected_state = expected_state
        self.current_state = current_state


def canonical_json(value: Any) -> str:
    """把同一份数据稳定地编码成完全相同的JSON文字。

    这份文字用于计算哈希和判断重复提交；NaN、无穷大等非标准JSON值会被
    拒绝，避免不同Python或数据库实现给出不同结果。
    """

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"值不是严格 JSON：{exc}") from exc


def canonical_json_sha256(value: Any) -> str:
    """对稳定编码后的JSON计算内容哈希，用来识别同一份事实。"""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def extract_gaussian_code(value: Any) -> str | None:
    """从 codegen 完成收据的有限结果包中读取实际 Gaussian route。"""

    if not isinstance(value, Mapping):
        return None
    candidates: list[Any] = [value]
    for key in ("result", "raw_output", "raw_tool_result"):
        nested = value.get(key)
        if isinstance(nested, Mapping):
            candidates.append(nested)
            if isinstance(nested.get("result"), Mapping):
                candidates.append(nested["result"])
    for candidate in candidates:
        route = candidate.get("gaussian_code") if isinstance(candidate, Mapping) else None
        if isinstance(route, str) and route.strip():
            return route.strip()
    return None


def empty_runtime_checkpoint() -> Dict[str, Any]:
    """创建一份尚未执行任何步骤的进度记录。

    所有游标和当前操作引用都从同一个初始值开始，恢复器无需猜测字段缺失究竟
    表示“尚未开始”还是“旧版本没有保存”。
    """

    return {
        "checkpoint_id": None,
        "checkpoint_index": 0,
        "checkpoint_manifest_ref": None,
        "current_operation_instance_id": None,
        "current_request_ref": None,
        "next_step_index": 0,
        "step_manifest_refs": [],
        # 每个 codegen step_id 只保存一份指向唯一 step manifest 的 route 投影；
        # manifest 和 raw/normalized artifact 仍是恢复时的权威收据。
        "gaussian_code_results": {},
    }


def require_exact_keys(
    value: Any,
    keys: Iterable[str],
    path: str,
    *,
    optional_fields: Iterable[str] = (),
) -> Mapping[str, Any]:
    """确认对象既没有漏掉必需字段，也没有混入含义不明的新字段。"""

    if not isinstance(value, dict):
        raise ContractValidationError(f"{path} 必须是 JSON object")
    expected = set(keys)
    allowed = expected | set(optional_fields)
    actual = set(value)
    if not expected <= actual or not actual <= allowed:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ContractValidationError(f"{path} 字段不匹配：missing={missing}, extra={extra}")
    return value


def require_nonempty_string(value: Any, path: str) -> str:
    """读取一个必填字符串，并拒绝空白内容。"""

    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{path} 必须是非空字符串")
    return value


def derive_hypothesis_compatibility_text(claims: Any) -> Dict[str, str]:
    """把有序原子主张确定性地投影成旧模块仍在读取的三段文字。

    主张及理由不会被截断、概括或重新排序。标签让旧的 Planner、比较器和结果
    展示仍能辨认每项内容，同时不会把多项科学结论压回一段含义不清的自由文本。
    """

    if not isinstance(claims, list) or not 1 <= len(claims) <= 4:
        raise ContractValidationError("hypothesis.claims 必须包含 1–4 项原子主张")
    statements: List[str] = []
    rationales: List[str] = []
    for index, raw_claim in enumerate(claims, start=1):
        claim = require_exact_keys(
            raw_claim,
            {"statement", "rationale"},
            f"hypothesis.claims[{index - 1}]",
        )
        statement = require_nonempty_string(
            claim["statement"],
            f"hypothesis.claims[{index - 1}].statement",
        )
        rationale = require_nonempty_string(
            claim["rationale"],
            f"hypothesis.claims[{index - 1}].rationale",
        )
        if statement != statement.strip() or rationale != rationale.strip():
            raise ContractValidationError("hypothesis.claims 的文字必须已经去除首尾空白")
        if len(statement) > 600:
            raise ContractValidationError("hypothesis claim.statement 最长为 600 字符")
        if len(rationale) > 800:
            raise ContractValidationError("hypothesis claim.rationale 最长为 800 字符")
        statements.append(statement)
        rationales.append(rationale)
    if len(set(statements)) != len(statements):
        raise ContractValidationError("同一 hypothesis 的 claim.statement 不得重复")

    core_claim = "\n".join(
        f"[c{index}] {statement}"
        for index, statement in enumerate(statements, start=1)
    )
    reasoning = "\n\n".join(
        f"[c{index}] {statement}\n[r{index}] {rationales[index - 1]}"
        for index, statement in enumerate(statements, start=1)
    )
    return {
        "core_claim": core_claim,
        "reasoning": reasoning,
        "detailed_reasoning": reasoning,
    }


def validate_verification_requirements(
    requirements: Any,
    *,
    path: str = "hypothesis.verification_requirements",
) -> List[str]:
    """校验验证需求的确定性结构边界并逐值复制。"""

    if not isinstance(requirements, list) or not 1 <= len(requirements) <= 4:
        raise ContractValidationError(f"{path} 必须包含 1–4 条自然语言文本")
    validated: List[str] = []
    for index, requirement in enumerate(requirements):
        if (
            not isinstance(requirement, str)
            or not requirement.strip()
            or requirement != requirement.strip()
            or len(requirement) > 800
        ):
            raise ContractValidationError(
                f"{path}[{index}] 必须是无首尾空白的非空文本"
            )
        validated.append(requirement)
    if len(set(validated)) != len(validated):
        raise ContractValidationError(f"{path} 不得包含逐字重复项")
    return validated


def validate_structured_hypothesis_claims(
    value: Any,
    *,
    path: str = "hypothesis",
    allow_legacy: bool = True,
) -> Dict[str, str]:
    """校验活动假设的两字段正文，并隔离读取旧版快照。

    新候选只保存 ``strategy_name`` 与完整 ``reasoning``；排名、身份等程序元数据
    可以由调用方附加。旧快照若仍包含 ``claims``、``core_claim`` 或
    ``detailed_reasoning``，仅在 ``allow_legacy`` 为真时逐字核对并返回兼容内容，
    不表示这些字段可以进入新的事件或模型输入。
    """

    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{path} 必须是 object")
    if "strategy_name" not in value:
        raise ContractValidationError(f"{path} 缺少正式假设字段：['strategy_name']")
    require_nonempty_string(value["strategy_name"], f"{path}.strategy_name")

    legacy_fields = {"claims", "core_claim", "detailed_reasoning"}
    if legacy_fields.intersection(value):
        if not allow_legacy:
            raise ContractValidationError(
                f"{path} 新协议不得包含旧版字段："
                f"{sorted(legacy_fields.intersection(value))}"
            )
        required = {"strategy_name", "core_claim", "reasoning"}
        missing = sorted(required - set(value))
        if missing:
            raise ContractValidationError(f"{path} 缺少旧版假设字段：{missing}")
        core_claim = require_nonempty_string(value["core_claim"], f"{path}.core_claim")
        reasoning = require_nonempty_string(value["reasoning"], f"{path}.reasoning")
        if core_claim != core_claim.strip() or reasoning != reasoning.strip():
            raise ContractValidationError(f"{path} 科学正文必须已经去除首尾空白")

        if "claims" in value:
            if "verification_requirements" in value:
                validate_verification_requirements(
                    value["verification_requirements"],
                    path=f"{path}.verification_requirements",
                )
            derived = derive_hypothesis_compatibility_text(value["claims"])
            fields_to_check = {"core_claim", "reasoning"}
            if "detailed_reasoning" in value:
                fields_to_check.add("detailed_reasoning")
            for field in fields_to_check:
                expected = derived[field]
                if value.get(field) != expected:
                    raise ContractValidationError(
                        f"{path}.{field} 与 claims 的确定性派生结果不一致"
                    )
            return derived

        if len(core_claim) > 600:
            raise ContractValidationError(f"{path}.core_claim 最长为 600 字符")
        if len(reasoning) > 3200:
            raise ContractValidationError(f"{path}.reasoning 最长为 3200 字符")
        return {
            "core_claim": core_claim,
            "reasoning": reasoning,
            # 该字段只为旧消费者提供读取兼容，不要求新对象重复保存。
            "detailed_reasoning": reasoning,
        }

    reasoning = require_nonempty_string(value.get("reasoning"), f"{path}.reasoning")
    if reasoning != reasoning.strip():
        raise ContractValidationError(f"{path} 科学正文必须已经去除首尾空白")
    if len(reasoning) > 3200:
        raise ContractValidationError(f"{path}.reasoning 最长为 3200 字符")
    return {
        "strategy_name": value["strategy_name"].strip(),
        "reasoning": reasoning,
    }


def require_json_integer(value: Any, path: str, minimum: int) -> int:
    """读取JSON整数；布尔值虽是Python整数子类，也不能冒充计数。"""

    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractValidationError(f"{path} 必须是大于等于 {minimum} 的 JSON integer")
    return value


def validate_task_config(task_config: Any) -> Dict[str, int]:
    """在创建任务前检查批次数量和预算，并返回可直接保存的副本。

    配置写入任务事实后不会在运行中改变，因此不合理组合必须在这里提前报错。
    """

    config = require_exact_keys(task_config, TASK_CONFIG_FIELDS, "task_config")
    result: Dict[str, int] = {}
    for field in sorted(TASK_CONFIG_FIELDS):
        minimum = 0 if field == "max_workflow_revisions_per_combination" else 1
        result[field] = require_json_integer(config[field], f"task_config.{field}", minimum)
    maximum = result["num_queries"] * result["num_hypotheses_per_query"]
    if result["top_n_workflows"] > maximum:
        raise ContractValidationError(
            "task_config.top_n_workflows 不能超过 num_queries * num_hypotheses_per_query"
        )
    return result


def validate_investigation_task_config(task_config: Any) -> Dict[str, Any]:
    """校验并返回 Investigation v2 的冻结任务配置。

    v2 与 legacy workflow 使用不同字段集合，避免恢复时把旧工作流修订预算误读为
    Investigation repair 预算。标准入口固定选择五个假设，调查并发必须在 1 到 5
    之间；repair 预算是任务级正整数配置，写入 TaskCreated 后随任务冻结。
    """

    config = require_exact_keys(
        task_config,
        INVESTIGATION_TASK_CONFIG_FIELDS,
        "investigation_task_config",
        optional_fields={
            INVESTIGATION_RUN_MODE_FIELD,
            INVESTIGATION_COMBINATION_COUNT_FIELD,
            INVESTIGATION_MODE_FIELD,
        },
    )
    result: Dict[str, Any] = {}
    for field in sorted(INVESTIGATION_TASK_CONFIG_FIELDS):
        result[field] = require_json_integer(
            config[field],
            f"investigation_task_config.{field}",
            1,
        )
    if INVESTIGATION_RUN_MODE_FIELD in config:
        run_mode = config[INVESTIGATION_RUN_MODE_FIELD]
        if run_mode not in INVESTIGATION_RUN_MODES:
            raise ContractValidationError(
                "investigation_task_config.run_mode 未注册"
            )
        result[INVESTIGATION_RUN_MODE_FIELD] = run_mode
    mode = config.get(INVESTIGATION_MODE_FIELD, "standard")
    if mode != "standard" and mode not in AUTHOR_MAIN_HYPOTHESIS_MODES:
        raise ContractValidationError("investigation_mode 未注册")
    if INVESTIGATION_COMBINATION_COUNT_FIELD in config:
        result[INVESTIGATION_COMBINATION_COUNT_FIELD] = require_json_integer(
            config[INVESTIGATION_COMBINATION_COUNT_FIELD],
            f"investigation_task_config.{INVESTIGATION_COMBINATION_COUNT_FIELD}",
            1,
        )
        if result[INVESTIGATION_COMBINATION_COUNT_FIELD] != result["top_n_hypotheses"]:
            raise ContractValidationError(
                "investigation_combination_count 必须等于 top_n_hypotheses"
            )
    expected_count = result.get(
        INVESTIGATION_COMBINATION_COUNT_FIELD,
        result["top_n_hypotheses"],
    )
    if mode == "standard" and result["top_n_hypotheses"] != 5:
        raise ContractValidationError(
            "standard investigation_v2 的 top_n_hypotheses 必须固定为 5"
        )
    if mode in AUTHOR_MAIN_HYPOTHESIS_MODES and (
        result["top_n_hypotheses"] != 1 or expected_count != 1
    ):
        raise ContractValidationError(
            "作者主假设模式必须固定为 1 个组合"
        )
    if expected_count not in {1, 5}:
        raise ContractValidationError(
            "investigation_combination_count 只能是标准 5 或作者实验 1"
        )
    if mode in AUTHOR_MAIN_HYPOTHESIS_MODES:
        if (
            result["num_queries"] != 1
            or result["num_hypotheses_per_query"] != 1
            or result["max_parallel_investigations"] != 1
            or result["max_hypothesis_batches"] != 1
        ):
            raise ContractValidationError(
                "作者主假设模式必须冻结单查询、单候选、单并发和单批次"
            )
    if not 1 <= result["max_parallel_investigations"] <= 5:
        raise ContractValidationError(
            "investigation_task_config.max_parallel_investigations 必须位于 1 到 5 之间"
        )
    candidate_capacity = (
        result["num_queries"] * result["num_hypotheses_per_query"]
    )
    if result["top_n_hypotheses"] > candidate_capacity:
        raise ContractValidationError(
            "investigation_task_config.top_n_hypotheses 不能超过"
            " num_queries * num_hypotheses_per_query"
        )
    if INVESTIGATION_MODE_FIELD in config:
        result[INVESTIGATION_MODE_FIELD] = mode
    return result


def get_investigation_combination_count(task_config: Mapping[str, Any]) -> int:
    """从冻结 v2 配置读取当前批次应覆盖的组合数。

    输入是已经由 TaskCreated 保存的任务配置；动作读取显式组合数，旧任务缺失时
    回落到 top_n_hypotheses，返回 1 到 5 的整数。配置类型、数值范围或两字段不一
    致时抛出 ``ContractValidationError``，不访问数据库、文件或 provider。
    """

    if not isinstance(task_config, Mapping):
        raise ContractValidationError("investigation_task_config 必须是 JSON object")
    mode = task_config.get(INVESTIGATION_MODE_FIELD, "standard")
    if mode != "standard" and mode not in AUTHOR_MAIN_HYPOTHESIS_MODES:
        raise ContractValidationError("investigation_mode 未注册")
    top_n = require_json_integer(
        task_config.get("top_n_hypotheses"),
        "investigation_task_config.top_n_hypotheses",
        1,
    )
    value = task_config.get(INVESTIGATION_COMBINATION_COUNT_FIELD, top_n)
    count = require_json_integer(
        value,
        f"investigation_task_config.{INVESTIGATION_COMBINATION_COUNT_FIELD}",
        1,
    )
    if mode == "standard" and top_n != 5:
        raise ContractValidationError(
            "standard investigation_v2 的组合数必须冻结为 5"
        )
    if mode in AUTHOR_MAIN_HYPOTHESIS_MODES and (top_n != 1 or count != 1):
        raise ContractValidationError(
            "作者主假设模式的组合数必须冻结为 1"
        )
    if count != top_n or count not in {1, 5}:
        raise ContractValidationError(
            "investigation_combination_count 必须等于 top_n_hypotheses 且只能是 1 或 5"
        )
    return count


def _default_model_configuration() -> Dict[str, Any]:
    """返回当前协议的“专家关闭”运行配置。"""

    disabled_role = {
        "model": None,
        "base_url": None,
        "api_key_env": "",
        "api_key_configured": False,
        "config_fingerprint": "0" * 64,
    }
    return {
        "enabled": False,
        "status": "disabled",
        "expert": dict(disabled_role),
        "fallback": dict(disabled_role),
        "general": dict(disabled_role),
    }


def _validate_current_expert_runtime_selection(value: Any) -> Dict[str, Any]:
    """校验任务级专家后端选择事实。

    输入是待持久化的选择映射；动作检查固定字段、公开 provider 信息和启动探测
    结果，不读取 tokenizer、profile 或精确 token 事实，也不修改输入。返回值是可
    继续使用的副本；字段缺失、额外字段或选择与探测不一致时抛出合同校验异常。
    """

    if not isinstance(value, dict):
        raise ContractValidationError("expert_backend_selection 必须是 object")
    required = {
        "schema_version",
        "status",
        "backend",
        "provider_role",
        "provider",
        "served_model",
        "probe",
        "fallback_reason",
        "failure_role",
    }
    if set(value) != required:
        raise ContractValidationError("runtime_selection 字段不符合固定契约")
    if value["schema_version"] != "expert-runtime-selection.v2":
        raise ContractValidationError("runtime_selection.schema_version 未注册")
    if value["status"] not in {"selected", "startup_failed", "disabled"}:
        raise ContractValidationError("runtime_selection.status 非法")
    if value["backend"] not in {
        "arche_chem_7b",
        "cluster_expert",
        "fallback",
        "startup_failed",
        "disabled",
    }:
        raise ContractValidationError("runtime_selection.backend 未注册")
    if value["provider_role"] not in {None, "expert", "cluster", "fallback"}:
        raise ContractValidationError("runtime_selection.provider_role 非法")
    if value["failure_role"] not in {None, "fallback", "general"}:
        raise ContractValidationError("runtime_selection.failure_role 非法")

    def validate_probe(probe: Any, path: str, *, allow_none: bool = False) -> None:
        if probe is None and allow_none:
            return
        if not isinstance(probe, dict) or set(probe) != {
            "connectivity",
            "authentication",
            "served_model",
            "strict_json",
            "reason_code",
        }:
            raise ContractValidationError(f"{path} 字段不完整")
        for field in ("connectivity", "authentication", "strict_json"):
            if not isinstance(probe[field], bool):
                raise ContractValidationError(f"{path}.{field} 必须是 boolean")
        if probe["served_model"] is not None and (
            not isinstance(probe["served_model"], str)
            or not probe["served_model"].strip()
        ):
            raise ContractValidationError(f"{path}.served_model 非法")
        if probe["reason_code"] not in {
            "ok",
            "provider_unconfigured",
            "provider_authentication_error",
            "provider_transport_error",
            "provider_protocol_error",
            "provider_model_rejected",
            "provider_invalid_json",
            "provider_probe_missing_served_model",
            "provider_probe_model_identity_mismatch",
        }:
            raise ContractValidationError(f"{path}.reason_code 未注册")
        if probe["reason_code"] == "ok":
            if not all(probe[field] for field in ("connectivity", "authentication", "strict_json")):
                raise ContractValidationError(f"{path} 的成功事实自相矛盾")
            if probe["served_model"] is None:
                raise ContractValidationError(f"{path}.served_model 缺失")
        elif probe["reason_code"] == "provider_unconfigured" and any(
            probe[field] for field in ("connectivity", "authentication", "strict_json")
        ):
            raise ContractValidationError(f"{path} 未配置事实自相矛盾")

    provider = value["provider"]
    if not isinstance(provider, dict) or set(provider) != {
        "model",
        "base_url",
        "api_key_env",
        "api_key_configured",
        "config_fingerprint",
    }:
        raise ContractValidationError("runtime_selection.provider 字段不完整")
    for field in ("model", "base_url"):
        if provider[field] is not None and (
            not isinstance(provider[field], str) or not provider[field].strip()
        ):
            raise ContractValidationError("runtime_selection.provider 身份字段非法")
    if not isinstance(provider["api_key_env"], str) or not isinstance(
        provider["api_key_configured"], bool
    ):
        raise ContractValidationError("runtime_selection.provider 认证标记非法")
    fingerprint = provider["config_fingerprint"]
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(char not in "0123456789abcdef" for char in fingerprint)
    ):
        raise ContractValidationError("runtime_selection.provider 指纹非法")
    if value["status"] == "disabled":
        if value["backend"] != "disabled" or value["provider_role"] is not None:
            raise ContractValidationError("disabled runtime_selection 身份字段非法")
        if provider != {} and provider != {
            "model": None,
            "base_url": None,
            "api_key_env": "",
            "api_key_configured": False,
            "config_fingerprint": "0" * 64,
        }:
            raise ContractValidationError("disabled runtime_selection 不得携带 provider")
        if value["served_model"] is not None or value["probe"] != {
            "expert": None,
            "fallback": None,
            "general": None,
        }:
            raise ContractValidationError("disabled runtime_selection 运行事实必须为空")
        if value["fallback_reason"] is not None or value["failure_role"] is not None:
            raise ContractValidationError("disabled runtime_selection 不得携带失败原因")
        return dict(value)
    if value["status"] == "selected":
        if value["backend"] not in {
            "arche_chem_7b",
            "cluster_expert",
            "fallback",
        }:
            raise ContractValidationError("selected runtime_selection backend 非法")
        expected_role = {
            "arche_chem_7b": "expert",
            "cluster_expert": "cluster",
            "fallback": "fallback",
        }[value["backend"]]
        if value["provider_role"] != expected_role:
            raise ContractValidationError("selected runtime_selection provider_role 与 backend 不一致")
        if not all(
            isinstance(provider[field], str) and provider[field].strip()
            for field in ("model", "base_url", "api_key_env")
        ) or provider["api_key_configured"] is not True:
            raise ContractValidationError("selected runtime_selection.provider 必须完整且已配置")
    else:
        if (
            value["backend"] != "startup_failed"
            or value["provider_role"] is not None
            or value["failure_role"] not in {"fallback", "general"}
        ):
            raise ContractValidationError("startup_failed runtime_selection 身份字段非法")
        require_nonempty_string(value["fallback_reason"], "runtime_selection.fallback_reason")

    probe = value["probe"]
    expected_probe_keys = {"expert", "fallback", "general"}
    if isinstance(probe, dict) and "cluster" in probe:
        expected_probe_keys.add("cluster")
    if not isinstance(probe, dict) or set(probe) != expected_probe_keys:
        raise ContractValidationError(
            "runtime_selection.probe 必须固定包含 expert/fallback/general，可选 cluster"
        )
    validate_probe(probe["expert"], "runtime_selection.probe.expert")
    validate_probe(probe["fallback"], "runtime_selection.probe.fallback", allow_none=True)
    validate_probe(probe["general"], "runtime_selection.probe.general")
    cluster_probe = probe.get("cluster")
    if "cluster" in probe:
        validate_probe(cluster_probe, "runtime_selection.probe.cluster", allow_none=True)
    expert_probe = probe["expert"]
    fallback_probe = probe["fallback"]
    general_probe = probe["general"]
    if value["status"] == "selected" and value["backend"] == "arche_chem_7b":
        if (
            expert_probe["reason_code"] != "ok"
            or not isinstance(general_probe, dict)
            or general_probe["reason_code"] != "ok"
            or fallback_probe is not None
        ):
            raise ContractValidationError("7B selected runtime_selection 探测事实不一致")
        if not model_identity_matches(provider["model"], value["served_model"]):
            raise ContractValidationError("selected runtime_selection.served_model 与 provider 不一致")
        if value["fallback_reason"] is not None or value["failure_role"] is not None:
            raise ContractValidationError("selected 7B 不应携带 fallback_reason")
    elif value["status"] == "selected" and value["backend"] == "cluster_expert":
        if (
            expert_probe["reason_code"] == "ok"
            or not isinstance(cluster_probe, dict)
            or cluster_probe["reason_code"] != "ok"
            or fallback_probe is not None
            or not isinstance(general_probe, dict)
            or general_probe["reason_code"] != "ok"
        ):
            raise ContractValidationError("cluster selected runtime_selection 探测事实不一致")
        if not model_identity_matches(provider["model"], value["served_model"]):
            raise ContractValidationError("selected runtime_selection.served_model 与 provider 不一致")
        if value["fallback_reason"] is not None or value["failure_role"] is not None:
            raise ContractValidationError("selected cluster 不应携带 fallback_reason")
    elif value["status"] == "selected":
        if (
            expert_probe["reason_code"] == "ok"
            or not isinstance(fallback_probe, dict)
            or fallback_probe["reason_code"] != "ok"
            or not isinstance(general_probe, dict)
            or general_probe["reason_code"] != "ok"
        ):
            raise ContractValidationError("Flash 回退 runtime_selection 探测事实不一致")
        if (
            not model_identity_matches(provider["model"], value["served_model"])
            or value["fallback_reason"]
            != (
                cluster_probe["reason_code"]
                if isinstance(cluster_probe, dict)
                else expert_probe["reason_code"]
            )
            or value["failure_role"] is not None
        ):
            raise ContractValidationError("Flash 回退 runtime_selection 身份事实不一致")
    else:
        # 即使某个候选后端可用，另一个必需后端失败也会让整个专家阶段无法启动。
        # failure_role 明确记录阻止任务启动的候选，避免把 Flash 失败误报为 Pro 失败。
        if value["failure_role"] == "fallback":
            if (
                not isinstance(fallback_probe, dict)
                or fallback_probe["reason_code"] == "ok"
            ):
                raise ContractValidationError("startup_failed fallback 探测事实不一致")
            if value["fallback_reason"] != fallback_probe["reason_code"]:
                raise ContractValidationError(
                    "startup_failed 失败原因与 fallback probe 不一致"
                )
        else:
            if not isinstance(general_probe, dict) or general_probe["reason_code"] == "ok":
                raise ContractValidationError("startup_failed general 探测事实不一致")
            if value["fallback_reason"] != general_probe["reason_code"]:
                raise ContractValidationError(
                    "startup_failed 失败原因与 general probe 不一致"
                )
    return dict(value)


def validate_expert_backend_selection(value: Any) -> Dict[str, Any]:
    """校验 TaskCreated 顶层专家后端选择事实。"""

    return _validate_current_expert_runtime_selection(value)


_MODEL_CONFIGURATION_ROLE_FIELDS = {
    "model",
    "base_url",
    "api_key_env",
    "api_key_configured",
    "config_fingerprint",
}


def _read_model_configuration_header(value: Any) -> tuple[bool, str, list[str]]:
    """读取模型配置的顶层开关、状态和角色集合。

    输入是待写入 TaskCreated 的模型配置；动作检查顶层字段和枚举，不访问外部
    provider；返回 ``enabled``、``status`` 和角色名列表，字段不完整时抛出合同错误。
    """

    if not isinstance(value, dict):
        raise ContractValidationError("model_configuration 必须是 JSON object")
    expected = {"enabled", "status", "expert", "fallback", "general"}
    if "cluster" in value:
        expected.add("cluster")
    extra = set(value) - expected
    missing = expected - set(value)
    if missing or extra:
        raise ContractValidationError(
            f"model_configuration 字段不匹配：missing={sorted(missing)}, extra={sorted(extra)}"
        )
    enabled = value["enabled"]
    if not isinstance(enabled, bool):
        raise ContractValidationError("model_configuration.enabled 必须是 boolean")
    status = value["status"]
    if not isinstance(status, str) or status not in {
        "disabled",
        "configured",
        "not_configured",
    }:
        raise ContractValidationError("model_configuration.status 非法")
    if not enabled and status != "disabled":
        raise ContractValidationError("关闭专家时 model_configuration.status 必须是 disabled")
    role_names = ["expert", "fallback", "general"]
    if "cluster" in value:
        role_names.append("cluster")
    return enabled, status, role_names


def _validate_model_configuration_role_field_types(
    role: Mapping[str, Any],
    role_name: str,
) -> None:
    """检查一个 provider 角色的公开字段类型和安全指纹格式。"""

    for field in ("model", "base_url"):
        if role[field] is not None and (
            not isinstance(role[field], str) or not role[field].strip()
        ):
            raise ContractValidationError(
                f"model_configuration.{role_name}.{field} 必须是非空字符串或 null"
            )
    if not isinstance(role["api_key_env"], str):
        raise ContractValidationError(
            f"model_configuration.{role_name}.api_key_env 必须是字符串"
        )
    if not isinstance(role["api_key_configured"], bool):
        raise ContractValidationError(
            f"model_configuration.{role_name}.api_key_configured 必须是 boolean"
        )
    fingerprint = role["config_fingerprint"]
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or fingerprint != fingerprint.lower()
        or any(char not in "0123456789abcdef" for char in fingerprint)
    ):
        raise ContractValidationError(
            f"model_configuration.{role_name}.config_fingerprint 必须是 64 位小写 SHA-256"
        )


def _validate_model_configuration_role(
    value: Any,
    role_name: str,
) -> Dict[str, Any]:
    """校验并复制一个 provider 角色的公开元数据。"""

    role = require_exact_keys(
        value,
        _MODEL_CONFIGURATION_ROLE_FIELDS,
        f"model_configuration.{role_name}",
    )
    _validate_model_configuration_role_field_types(role, role_name)
    return dict(role)


def _model_configuration_role_is_ready(role: Mapping[str, Any]) -> bool:
    """判断角色是否具备运行所需的模型、地址和认证存在性标记。"""

    return bool(
        role["model"]
        and role["base_url"]
        and role["api_key_env"]
        and role["api_key_configured"]
    )


def _validate_model_configuration_status(
    enabled: bool,
    status: str,
    role_configured: Mapping[str, bool],
) -> None:
    """校验模型开关、状态和可运行角色之间的关系。"""

    if not enabled:
        return
    specialist_ready = (
        role_configured["expert"]
        or role_configured.get("cluster", False)
        or role_configured["fallback"]
    )
    expected_status = (
        "configured"
        if role_configured["general"] and specialist_ready
        else "not_configured"
    )
    if status != expected_status:
        raise ContractValidationError(
            "model_configuration.status 与可运行的专家/通用 provider 配置不一致"
        )


def _disabled_model_configuration_role() -> Dict[str, Any]:
    """返回专家关闭时每个 provider 角色必须使用的固定空配置。"""

    return {
        "model": None,
        "base_url": None,
        "api_key_env": "",
        "api_key_configured": False,
        "config_fingerprint": "0" * 64,
    }


def _validate_disabled_model_configuration_roles(
    normalized: Mapping[str, Any],
    role_names: Sequence[str],
) -> None:
    """确认关闭专家时所有 provider 角色都没有残留运行配置。"""

    disabled_role = _disabled_model_configuration_role()
    if any(normalized[name] != disabled_role for name in role_names):
        raise ContractValidationError(
            "关闭专家时三个 provider 角色必须使用固定空配置"
        )


def validate_model_configuration(value: Any) -> Dict[str, Any]:
    """校验可审计的专家模型运行元数据，不接受任何认证正文。

    输入是待写入 TaskCreated 的 JSON 对象。动作校验并规范化专家开关、本地专家、
    可选集群专家、Flash 回退和 General Pro 的公开配置指纹；副作用是无，不访问
    provider、文件或数据库。返回不含认证正文的规范化 mapping；字段缺失、多余、类型
    错误、指纹非法或配置状态不一致时抛出 ``ContractValidationError``。
    """

    enabled, status, role_names = _read_model_configuration_header(value)
    normalized: Dict[str, Any] = {"enabled": enabled, "status": status}
    for role_name in role_names:
        normalized[role_name] = _validate_model_configuration_role(
            value[role_name], role_name
        )
    role_configured = {
        name: _model_configuration_role_is_ready(normalized[name])
        for name in role_names
    }
    _validate_model_configuration_status(enabled, status, role_configured)
    if not enabled:
        _validate_disabled_model_configuration_roles(normalized, role_names)
    return normalized


def validate_task_relative_path(value: Any, path: str) -> str:
    """确认保存的是任务目录内的相对路径，而不是某台机器的绝对位置。

    父目录跳转和反斜杠会造成跨系统解释差异，因此统一拒绝。
    """

    text = require_nonempty_string(value, path)
    if "\x00" in text or "\\" in text:
        raise ContractValidationError(f"{path} 必须使用 POSIX 分隔符")
    posix = PurePosixPath(text)
    if posix.is_absolute() or ".." in posix.parts or "." in posix.parts:
        raise ContractValidationError(f"{path} 必须是安全的 task 相对路径")
    if str(posix) != text:
        raise ContractValidationError(f"{path} 必须是规范化 POSIX 路径")
    return text


def validate_stage_architecture_version(value: Any) -> str:
    """校验任务冻结的阶段架构版本。

    输入是 TaskCreated 中的公开枚举值；动作不读取环境变量，也不接受任意新字符串。
    返回值可直接写入事件和任务当前索引，未知值会立即抛出合同错误。
    """

    if not isinstance(value, str) or value not in STAGE_ARCHITECTURE_VERSIONS:
        allowed = "、".join(sorted(STAGE_ARCHITECTURE_VERSIONS))
        raise ContractValidationError(
            f"stage_architecture_version 必须是已注册枚举：{allowed}"
        )
    return value


_PRIVATE_INVESTIGATION_TEXT_RE = re.compile(
    r"(?:\b(?:access[_ -]?token|api[_ -]?key|secret|password|credential|transcript|"
    r"private[_ -]?key|authorization)\b)"
    r"|(?:file://|[A-Za-z]:[\\/]|^/|^//|(?:^|\s)\.\.?/)"
    r"|(?:/(?:root|home|tmp|var|mnt|workspace)(?:/|$))",
    re.IGNORECASE,
)
_SCOPE_SUBMISSION_OPERATION_ID_RE = re.compile(
    r"^scope-submission-[0-9a-f]{64}$"
)


def _validate_investigation_text(value: Any, path: str) -> str:
    """校验调查正文，拒绝凭据、transcript 和机器绝对路径。"""

    text = require_nonempty_string(value, path)
    if _PRIVATE_INVESTIGATION_TEXT_RE.search(text):
        raise ContractValidationError(f"{path} 不能包含私有会话、凭据或绝对路径")
    return text


def _validate_investigation_reference(value: Any, path: str) -> str:
    """校验作业、操作和 artifact 的稳定引用，不保存本机路径。"""

    text = _validate_investigation_text(value, path)
    parts = text.split("/")
    if "\x00" in text or "\\" in text or any(
        part in {"", ".", ".."} for part in parts
    ):
        raise ContractValidationError(f"{path} 必须是安全的稳定引用")
    if PurePosixPath(text).is_absolute() or text.startswith("//"):
        raise ContractValidationError(f"{path} 不能是绝对路径")
    return text


def _validate_investigation_reference_list(value: Any, path: str) -> List[str]:
    """校验稳定引用数组并拒绝重复项。

    输入是作业、操作或 artifact 的稳定引用数组；动作拒绝空值、重复项、凭据和
    本机路径；返回规范化字符串列表，不写入任何状态，失败时抛出合同错误。
    """

    if not isinstance(value, (list, tuple)):
        raise ContractValidationError(f"{path} 必须是字符串数组")
    result = [
        _validate_investigation_reference(item, f"{path}[{index}]")
        for index, item in enumerate(value)
    ]
    if len(result) != len(set(result)):
        raise ContractValidationError(f"{path} 不能包含重复引用")
    return result


def _validate_scope_submission_operation_id(value: Any, path: str) -> str:
    """校验由 MiniChem scope 和 submission key 生成的提交 operation 身份。

    输入是待保存字段及其错误路径；函数只检查固定前缀和 64 位小写十六进制摘要，
    不读取服务状态或修改持久数据。返回可写入证据事件的原字符串；字段类型、格式
    或摘要长度不符合协议时抛出 ``ContractValidationError``。
    """

    operation_id = _validate_investigation_reference(value, path)
    if _SCOPE_SUBMISSION_OPERATION_ID_RE.fullmatch(operation_id) is None:
        raise ContractValidationError(
            f"{path} 必须是规范的 scope submission operation"
        )
    return operation_id


def _validate_investigation_identity(value: Any, path: str) -> str:
    """校验内部恢复身份；它可写入内部列，但不会进入默认公开当前视图。"""

    return _validate_investigation_reference(value, path)


def _require_turn_termination_kind_fields(
    value: Mapping[str, Any],
) -> None:
    """按 lifecycle kind 要求 closure 的真实 owner 身份字段。

    输入是已经完成通用类型校验的 closure mapping；动作确认 initial、repair、review
    各自拥有足够的 session/request/thread 身份，且 Review thread 不被伪造。返回值为空；
    缺少字段或跨 kind 身份不一致时抛出合同错误，防止调用方把后续 projection 校验
    当作公开接口的唯一保护。
    """

    kind = value["kind"]
    required_by_kind = {
        "initial": ("combination_id", "session_id", "codex_thread_id", "scope_id"),
        "repair": ("combination_id", "request_event_ref", "codex_thread_id", "scope_id"),
        "review": ("review_turn_id", "review_thread_id", "codex_thread_id"),
    }
    missing = [field for field in required_by_kind[kind] if not value.get(field)]
    if missing:
        raise ContractValidationError(
            f"{kind} 终止事实缺少 owner 字段: {', '.join(missing)}"
        )
    if kind == "review" and value["review_thread_id"] != value["codex_thread_id"]:
        raise ContractValidationError("review 终止事实的 review_thread_id 必须等于 codex_thread_id")


def _normalize_turn_attempt_fields(value: Dict[str, Any], *, path: str) -> None:
    """规范 turn 启动事实的 attempt 字段并为旧事件提供保守默认值。

    输入是已完成基础字段校验的 Started payload；动作把缺失的旧 v8 字段回填为
    ``attempt_number=1``、``attempt_status=in_flight`` 和旧 operation 的稳定 logical
    identity，不生成 request digest 或其他科学事实。返回值为空；编号、状态、身份或
    previous terminal 引用不合法时抛出合同错误，调用方事务不会因此留下半条事件。
    """

    attempt_number = value.get("attempt_number", 1)
    if (
        isinstance(attempt_number, bool)
        or not isinstance(attempt_number, int)
        or attempt_number < 1
    ):
        raise ContractValidationError(f"{path}.attempt_number 必须是正整数")
    value["attempt_number"] = attempt_number
    attempt_status = value.get("attempt_status", "in_flight")
    if attempt_status != "in_flight":
        raise ContractValidationError(f"{path}.attempt_status 必须为 in_flight")
    value["attempt_status"] = attempt_status
    logical_operation = value.get(
        "logical_operation_instance_id", value.get("operation_instance_id")
    )
    value["logical_operation_instance_id"] = _validate_investigation_identity(
        logical_operation, f"{path}.logical_operation_instance_id"
    )
    legacy_previous = value.pop("previous_attempt_terminal_event_ref", None)
    canonical_previous = value.get("previous_attempt_closure_event_ref")
    if (
        legacy_previous is not None
        and canonical_previous is not None
        and legacy_previous != canonical_previous
    ):
        raise ContractValidationError(
            f"{path}.previous attempt closure 引用重复且不一致"
        )
    previous = canonical_previous if canonical_previous is not None else legacy_previous
    if previous is not None:
        value["previous_attempt_closure_event_ref"] = _validate_investigation_reference(
            previous, f"{path}.previous_attempt_closure_event_ref"
        )


def validate_investigation_turn_attempt_termination_payload(
    payload: Any,
) -> Dict[str, Any]:
    """校验一次 provider turn 的明确终止事实。

    输入是服务从被引用 Started 事件逐值冻结的 attempt 身份；动作只校验终止状态、
    kind、身份和协议版本，返回可追加的 immutable closure 正文。它不读取 provider、
    transcript 或路径；缺字段、状态错误或协议漂移时抛出合同错误。
    """

    value = require_exact_keys(
        payload,
        {
            "task_id",
            "turn_started_event_ref",
            "terminal_status",
            "kind",
            "batch_id",
            "turn_id",
            "attempt_number",
            "logical_operation_instance_id",
            "attempt_operation_instance_id",
            "investigation_protocol_version",
        },
        "InvestigationTurnAttemptTerminated.payload",
        optional_fields={
            "combination_id",
            "session_id",
            "request_event_ref",
            "review_turn_id",
            "review_thread_id",
            "codex_thread_id",
            "scope_id",
            "previous_attempt_closure_event_ref",
            "previous_attempt_terminal_event_ref",
        },
    )
    for field in (
        "task_id",
        "turn_started_event_ref",
        "batch_id",
        "turn_id",
        "logical_operation_instance_id",
        "attempt_operation_instance_id",
    ):
        value[field] = _validate_investigation_identity(value[field], field)
    if value["kind"] not in {"initial", "repair", "review"}:
        raise ContractValidationError("Investigation turn termination kind 非法")
    if "combination_id" in value and value["combination_id"] is not None:
        value["combination_id"] = _validate_investigation_identity(
            value["combination_id"], "combination_id"
        )
    if value["kind"] in {"initial", "repair"} and not value.get("combination_id"):
        raise ContractValidationError("initial/repair 终止事实必须包含真实 combination_id")
    if value["kind"] == "review" and value.get("combination_id") is not None:
        raise ContractValidationError("review 终止事实不得伪造 combination_id")
    if value["terminal_status"] not in {"failed", "interrupted"}:
        raise ContractValidationError("terminal_status 只能是 failed 或 interrupted")
    if (
        isinstance(value["attempt_number"], bool)
        or not isinstance(value["attempt_number"], int)
        or value["attempt_number"] < 1
    ):
        raise ContractValidationError("attempt_number 必须是正整数")
    for field in ("session_id", "request_event_ref", "review_turn_id", "review_thread_id", "codex_thread_id", "scope_id"):
        if field in value and value[field] is not None:
            value[field] = _validate_investigation_identity(value[field], field)
    _require_turn_termination_kind_fields(value)
    legacy_previous = value.pop("previous_attempt_terminal_event_ref", None)
    canonical_previous = value.get("previous_attempt_closure_event_ref")
    if (
        legacy_previous is not None
        and canonical_previous is not None
        and legacy_previous != canonical_previous
    ):
        raise ContractValidationError(
            "终止事实的 previous closure 引用重复且不一致"
        )
    previous = canonical_previous if canonical_previous is not None else legacy_previous
    if previous is not None:
        value["previous_attempt_closure_event_ref"] = _validate_investigation_reference(
            previous, "previous_attempt_closure_event_ref"
        )
    value["investigation_protocol_version"] = _validate_investigation_protocol_version(
        value["investigation_protocol_version"], "investigation_protocol_version"
    )
    canonical_json(value)
    return dict(value)


def validate_investigation_legacy_waiting_resume_authorization_payload(
    payload: Any,
) -> Dict[str, Any]:
    """校验旧 waiting sidecar 的一次性恢复授权事实。

    输入必须由恢复装配层从同一 v1 sidecar、session、Started/closure 历史和已撤销
    scope 审计记录组合而成；动作只检查安全身份、摘要和固定 interrupted 条件，不
    读取 provider 或凭据。返回值是可追加的不可变授权正文；缺字段、携带 token、
    operation 格式错误或协议漂移时抛出合同错误。
    """

    value = require_exact_keys(
        payload,
        {
            "task_id",
            "batch_id",
            "combination_id",
            "session_id",
            "session_event_ref",
            "kind",
            "codex_thread_id",
            "workspace_ref_digest",
            "physical_workspace_digest",
            "scope_id",
            "turn_started_event_ref",
            "turn_id",
            "attempt_number",
            "logical_operation_instance_id",
            "attempt_operation_instance_id",
            "terminal_event_ref",
            "terminal_status",
            "job_id",
            "submission_operation_id",
            "submission_key",
            "submission_request_digest",
            "job_family",
            "action_id",
            "backend_id",
            "job_type",
            "runtime",
            "sidecar_version",
            "sidecar_record_digest",
            "sidecar_record_auth_digest",
            "previous_generation_hash",
            "investigation_protocol_version",
        },
        "InvestigationLegacyWaitingResumeAuthorized.payload",
        optional_fields={
            "software_id",
            "operation_fingerprint",
            "repair_request_event_ref",
        },
    )
    for field in (
        "task_id",
        "batch_id",
        "combination_id",
        "session_id",
        "session_event_ref",
        "codex_thread_id",
        "scope_id",
        "turn_started_event_ref",
        "turn_id",
        "logical_operation_instance_id",
        "attempt_operation_instance_id",
        "terminal_event_ref",
        "job_id",
        "submission_key",
        "job_family",
        "action_id",
        "backend_id",
        "job_type",
        "runtime",
    ):
        value[field] = _validate_investigation_identity(value[field], field)
    if value["kind"] not in {"initial", "repair"}:
        raise ContractValidationError("legacy waiting authorization.kind 非法")
    if value["terminal_status"] != "interrupted":
        raise ContractValidationError(
            "legacy waiting authorization 只能绑定 terminal_interrupted closure"
        )
    if (
        isinstance(value["attempt_number"], bool)
        or not isinstance(value["attempt_number"], int)
        or value["attempt_number"] < 1
    ):
        raise ContractValidationError("legacy waiting authorization.attempt_number 必须是正整数")
    # 该历史授权合同必须自含格式规则，不能依赖证据校验模块的并行实现；这里只
    # 接受 scope-submission- 加 64 位小写摘要的稳定提交身份。
    value["submission_operation_id"] = _validate_investigation_reference(
        value["submission_operation_id"], "submission_operation_id"
    )
    if re.fullmatch(
        r"scope-submission-[0-9a-f]{64}", value["submission_operation_id"]
    ) is None:
        raise ContractValidationError(
            "submission_operation_id 必须是规范的 scope submission operation"
        )
    for field in (
        "workspace_ref_digest",
        "physical_workspace_digest",
        "submission_request_digest",
        "sidecar_record_digest",
        "sidecar_record_auth_digest",
        "previous_generation_hash",
    ):
        value[field] = _validate_investigation_digest(value[field], field)
    if value["sidecar_version"] != 1:
        raise ContractValidationError("legacy waiting authorization.sidecar_version 必须为 1")
    for field in ("software_id",):
        if field in value and value[field] is not None:
            value[field] = _validate_investigation_identity(value[field], field)
    if "operation_fingerprint" in value and value["operation_fingerprint"] is not None:
        value["operation_fingerprint"] = _validate_investigation_digest(
            value["operation_fingerprint"], "operation_fingerprint"
        )
    if "repair_request_event_ref" in value and value["repair_request_event_ref"] is not None:
        value["repair_request_event_ref"] = _validate_investigation_identity(
            value["repair_request_event_ref"], "repair_request_event_ref"
        )
    if value["kind"] == "repair" and not value.get("repair_request_event_ref"):
        raise ContractValidationError(
            "legacy repair waiting authorization 缺少 request identity"
        )
    value["investigation_protocol_version"] = _validate_investigation_protocol_version(
        value["investigation_protocol_version"],
        "investigation_protocol_version",
    )
    canonical_json(value)
    return dict(value)


def validate_investigation_incomplete_v2_waiting_resume_authorization_payload(
    payload: Any,
) -> Dict[str, Any]:
    """校验残缺 v2 waiting sidecar 的一次性显式恢复授权事实。

    输入必须来自恢复装配层，包含当前 v2 sidecar 的两个 sealed 摘要、旧
    legacy 授权根事件、根 closure 到当前 Started 的完整 attempt 链，以及当前
    作业、工作区、scope 代次和执行回执摘要。动作只检查字段形状、稳定身份和
    SHA-256；不会读取 provider、sidecar 或凭据，也不会把任何可变当前状态写入
    payload。attempt 链的事件存在性、逐事件身份和 closure 状态由
    ``WorkingMemoryService`` 在同一事务中再次核对。

    ``attempt_chain`` 从旧授权根 closure 之后的第一个 Started 开始，到当前
    in-flight Started 结束。每个中间 Started 必须有 closure，最后一项的
    ``closure_event_ref`` 必须为 ``None``。
    """

    value = require_exact_keys(
        payload,
        {
            "task_id",
            "batch_id",
            "combination_id",
            "session_id",
            "session_event_ref",
            "kind",
            "codex_thread_id",
            "scope_id",
            "current_turn_started_event_ref",
            "current_turn_id",
            "current_attempt_number",
            "current_logical_operation_instance_id",
            "current_attempt_operation_instance_id",
            "root_attempt_closure_event_ref",
            "legacy_authorization_event_ref",
            "attempt_chain",
            "sidecar_version",
            "sidecar_record_digest",
            "sidecar_record_auth_digest",
            "job_id",
            "submission_operation_id",
            "submission_key",
            "submission_request_digest",
            "job_family",
            "logical_workspace_digest",
            "physical_workspace_digest",
            "active_scope_generation_digest",
            "execution_receipt_digest",
            "investigation_protocol_version",
        },
        "InvestigationIncompleteV2WaitingResumeAuthorized.payload",
        optional_fields={
            "action_id",
            "backend_id",
            "job_type",
            "runtime",
            "software_id",
            "operation_fingerprint",
            "repair_request_event_ref",
        },
    )
    for field in (
        "task_id",
        "batch_id",
        "combination_id",
        "session_id",
        "session_event_ref",
        "codex_thread_id",
        "scope_id",
        "current_turn_started_event_ref",
        "current_turn_id",
        "current_logical_operation_instance_id",
        "current_attempt_operation_instance_id",
        "root_attempt_closure_event_ref",
        "legacy_authorization_event_ref",
        "job_id",
        "submission_operation_id",
        "submission_key",
        "job_family",
    ):
        value[field] = _validate_investigation_identity(value[field], field)
    if value["kind"] not in {"initial", "repair"}:
        raise ContractValidationError(
            "incomplete v2 waiting authorization.kind 必须为 initial 或 repair"
        )
    for field in (
        "logical_workspace_digest",
        "physical_workspace_digest",
        "active_scope_generation_digest",
        "execution_receipt_digest",
        "sidecar_record_digest",
        "sidecar_record_auth_digest",
        "submission_request_digest",
    ):
        value[field] = _validate_investigation_digest(value[field], field)
    if (
        isinstance(value["sidecar_version"], bool)
        or not isinstance(value["sidecar_version"], int)
        or value["sidecar_version"] != 2
    ):
        raise ContractValidationError(
            "incomplete v2 waiting authorization.sidecar_version 必须为 2"
        )
    for field in (
        "current_attempt_number",
    ):
        if (
            isinstance(value[field], bool)
            or not isinstance(value[field], int)
            or value[field] < 1
        ):
            raise ContractValidationError(
                f"incomplete v2 waiting authorization.{field} 必须是正整数"
            )
    raw_chain = value["attempt_chain"]
    if not isinstance(raw_chain, list) or not raw_chain:
        raise ContractValidationError(
            "incomplete v2 waiting authorization.attempt_chain 必须是非空数组"
        )
    chain: list[Dict[str, Any]] = []
    for index, raw_item in enumerate(raw_chain):
        item = require_exact_keys(
            raw_item,
            {"attempt_number", "started_event_ref", "closure_event_ref"},
            f"InvestigationIncompleteV2WaitingResumeAuthorized.payload.attempt_chain[{index}]",
        )
        if (
            isinstance(item["attempt_number"], bool)
            or not isinstance(item["attempt_number"], int)
            or item["attempt_number"] < 1
        ):
            raise ContractValidationError(
                f"attempt_chain[{index}].attempt_number 必须是正整数"
            )
        normalized_item: Dict[str, Any] = {
            "attempt_number": item["attempt_number"],
            "started_event_ref": _validate_investigation_identity(
                item["started_event_ref"],
                f"attempt_chain[{index}].started_event_ref",
            ),
            "closure_event_ref": None,
        }
        closure_ref = item["closure_event_ref"]
        if closure_ref is not None:
            normalized_item["closure_event_ref"] = _validate_investigation_identity(
                closure_ref,
                f"attempt_chain[{index}].closure_event_ref",
            )
        chain.append(normalized_item)
    for index in range(1, len(chain)):
        if chain[index]["attempt_number"] != chain[index - 1]["attempt_number"] + 1:
            raise ContractValidationError(
                "incomplete v2 waiting authorization.attempt_chain 的 attempt number 不连续"
            )
    if any(item["closure_event_ref"] is None for item in chain[:-1]):
        raise ContractValidationError(
            "incomplete v2 waiting authorization.attempt_chain 的中间 attempt 缺少 closure"
        )
    if chain[-1]["closure_event_ref"] is not None:
        raise ContractValidationError(
            "incomplete v2 waiting authorization.attempt_chain 当前 Started 不得已有 closure"
        )
    if chain[-1]["started_event_ref"] != value["current_turn_started_event_ref"]:
        raise ContractValidationError(
            "incomplete v2 waiting authorization 当前 Started 不在 attempt_chain 末项"
        )
    if chain[-1]["attempt_number"] != value["current_attempt_number"]:
        raise ContractValidationError(
            "incomplete v2 waiting authorization 当前 attempt number 与 attempt_chain 不一致"
        )
    value["attempt_chain"] = chain
    for field in (
        "action_id",
        "backend_id",
        "job_type",
        "runtime",
        "software_id",
        "repair_request_event_ref",
    ):
        if field in value and value[field] is not None:
            value[field] = _validate_investigation_identity(value[field], field)
    if "operation_fingerprint" in value and value["operation_fingerprint"] is not None:
        value["operation_fingerprint"] = _validate_investigation_digest(
            value["operation_fingerprint"], "operation_fingerprint"
        )
    if value["kind"] == "repair" and not value.get("repair_request_event_ref"):
        raise ContractValidationError(
            "incomplete v2 repair waiting authorization 缺少 request identity"
        )
    value["investigation_protocol_version"] = _validate_investigation_protocol_version(
        value["investigation_protocol_version"],
        "investigation_protocol_version",
    )
    canonical_json(value)
    return dict(value)


def _validate_investigation_digest(value: Any, path: str) -> str:
    """校验证据适配器生成的不可逆摘要，不保存 receipt 正文或密钥。"""

    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ContractValidationError(f"{path} 必须是 64 位小写 SHA-256")
    return value


def _require_investigation_evidence_record_array(
    value: Any,
    path: str,
    *,
    allow_empty: bool = False,
) -> list[Any]:
    """确认作业摘要输入是数组，并按合同检查是否允许为空。

    输入是一个待核验的 JSON-like 数组；动作按 ``allow_empty`` 检查数组类型和空值
    约束，返回原数组供后续逐项规范化。输入不符合摘要合同会抛出合同错误，不写入
    数据库或修改调用方。
    """

    if not isinstance(value, (list, tuple)) or (not allow_empty and not value):
        expected = "摘要数组" if allow_empty else "非空摘要数组"
        raise ContractValidationError(f"{path} 必须是{expected}")
    return list(value)


def _validate_investigation_evidence_record_identities(
    item: Dict[str, Any],
    *,
    fields: set[str],
    path: str,
) -> None:
    """校验摘要记录中的作业、操作和 artifact 身份字段。

    输入是已确认字段集合的单条摘要字典；动作在内存中规范化稳定身份，不访问外部
    服务或路径；返回值为空，身份含秘密、空值或绝对路径时抛出合同错误。
    """

    for field in fields:
        if field.endswith("_id") or field in {"operation_id", "job_id", "artifact_id"}:
            item[field] = _validate_investigation_identity(
                item[field], f"{path}.{field}"
            )


def _validate_investigation_evidence_record_digests(
    item: Dict[str, Any],
    *,
    fields: set[str],
    path: str,
) -> None:
    """校验摘要记录中的不可逆 digest 字段。

    输入是已确认字段集合的单条摘要字典；动作只检查每个 digest 的 SHA-256 形状，
    返回值为空，格式错误时抛出合同错误，不保存原始收据内容。
    """

    for field in fields:
        if field.endswith("_digest"):
            item[field] = _validate_investigation_digest(item[field], f"{path}.{field}")


def _validate_investigation_evidence_record_status(
    item: Mapping[str, Any],
    *,
    status_field: str | None,
    path: str,
) -> None:
    """校验摘要记录的可选终态字段。

    输入是单条摘要字典和状态字段名；动作只接受已结束状态，返回值为空，状态缺失、
    类型错误或仍在运行时抛出合同错误，不修改输入。
    """

    if status_field is None:
        return
    status = item[status_field]
    if not isinstance(status, str):
        raise ContractValidationError(f"{path}.{status_field} 必须是字符串枚举")
    if status not in {"completed", "failed", "cancelled", "not_executed"}:
        raise ContractValidationError(f"{path}.{status_field} 必须是已结束状态")


def _validate_investigation_evidence_record(
    raw: Any,
    *,
    index: int,
    path: str,
    fields: set[str],
    status_field: str | None,
    optional_fields: set[str] | None = None,
) -> Dict[str, Any]:
    """规范化一条作业、操作或 artifact 摘要记录。

    输入是数组中的原始记录；动作确认精确字段、稳定身份、digest 和终态，返回新的
    字典副本。字段错误或含不安全内容时抛出合同错误，不产生数据库副作用。
    """

    item = dict(
        require_exact_keys(
            raw,
            fields,
            f"{path}[{index}]",
            optional_fields=optional_fields or (),
        )
    )
    item_path = f"{path}[{index}]"
    _validate_investigation_evidence_record_identities(
        item, fields=set(item), path=item_path
    )
    _validate_investigation_evidence_record_digests(
        item, fields=set(item), path=item_path
    )
    _validate_investigation_evidence_record_status(
        item, status_field=status_field, path=item_path
    )
    return item


def _ensure_investigation_evidence_record_identities_unique(
    records: Sequence[Mapping[str, Any]],
    path: str,
) -> None:
    """确认摘要数组中的作业、操作或 artifact 身份没有重复。

    输入是已经规范化的摘要记录；动作读取每条记录的首个稳定身份，返回值为空，
    重复身份时抛出合同错误，不修改记录。
    """

    identifiers = [
        next(
            item[field]
            for field in ("job_id", "artifact_id", "operation_id")
            if field in item
        )
        for item in records
    ]
    if len(identifiers) != len(set(identifiers)):
        raise ContractValidationError(f"{path} 的身份必须唯一")


def _validate_investigation_evidence_records(
    value: Any,
    *,
    path: str,
    fields: set[str],
    status_field: str | None = None,
    optional_fields: set[str] | None = None,
    allow_empty: bool = False,
) -> List[Dict[str, Any]]:
    """校验一组不含路径和私密正文的作业摘要，并返回规范化记录。"""

    records = _require_investigation_evidence_record_array(
        value,
        path,
        allow_empty=allow_empty,
    )
    result = [
        _validate_investigation_evidence_record(
            raw,
            index=index,
            path=path,
            fields=fields,
            status_field=status_field,
            optional_fields=optional_fields,
        )
        for index, raw in enumerate(records)
    ]
    _ensure_investigation_evidence_record_identities_unique(result, path)
    return result


def validate_investigation_evidence_payload(payload: Any) -> Dict[str, Any]:
    """校验上层权威适配器的安全证据摘要。

    输入是已经由 MiniChem/作业监督器核对后的摘要对象；动作绑定 task、batch、
    combination、thread、scope、turn，并要求所有 job/operation 都处于终态且每项
    都有摘要。旧 v1 摘要继续把 job/artifact 直接绑定 typed operation；v2 摘要允许
    没有 typed operation 的 job-only 来源保留空的 operations 和 artifacts 数组，仍把
    有内容的 job/artifact 摘要完整核对；同时把 artifact 唯一绑定到 producer job
    及该 job 的提交 operation，不能把提交 operation
    混入 Result 的 typed operation 清单。返回值可作为
    ``investigation_evidence_verified`` 事件正文；v2 对没有 scope submission 身份的
    native job 允许省略 ``submission_operation_id``，但 artifact 仍只能绑定到有该身份
    的 producer job。普通字符串、receipt 原文、token、
    私有 transcript 或绝对路径都会失败，方法不访问外部服务。
    """

    value = require_exact_keys(
        payload,
        {
            "task_id",
            "batch_id",
            "combination_id",
            "codex_thread_id",
            "scope_id",
            "turn_id",
            "verification_object_id",
            "verifier_id",
            "verifier_version",
            "verification_digest",
            "jobs",
            "operations",
            "artifacts",
            "operation_instance_id",
        },
        "InvestigationEvidenceVerified.payload",
        optional_fields={"reference_binding_version"},
    )
    for field in (
        "task_id",
        "batch_id",
        "combination_id",
        "codex_thread_id",
        "scope_id",
        "turn_id",
        "verification_object_id",
        "verifier_id",
        "verifier_version",
        "operation_instance_id",
    ):
        value[field] = _validate_investigation_identity(value[field], field)
    value["verification_digest"] = _validate_investigation_digest(
        value["verification_digest"], "verification_digest"
    )
    binding_version = value.get("reference_binding_version")
    value["operations"] = _validate_investigation_evidence_records(
        value["operations"],
        path="operations",
        fields={"operation_id", "terminal_status", "operation_digest"},
        status_field="terminal_status",
        allow_empty=binding_version == "producer-job-v2",
    )
    if binding_version is None:
        value["jobs"] = _validate_investigation_evidence_records(
            value["jobs"],
            path="jobs",
            fields={"job_id", "operation_id", "terminal_status", "receipt_digest"},
            status_field="terminal_status",
        )
        value["artifacts"] = _validate_investigation_evidence_records(
            value["artifacts"],
            path="artifacts",
            fields={"artifact_id", "operation_id", "artifact_digest"},
        )
        operation_ids = {item["operation_id"] for item in value["operations"]}
        if any(
            item["operation_id"] not in operation_ids
            for item in value["jobs"] + value["artifacts"]
        ):
            raise ContractValidationError(
                "v1 证据摘要的 job/artifact 必须绑定已验证 typed operation"
            )
    else:
        binding_version = _validate_investigation_identity(
            binding_version, "reference_binding_version"
        )
        if binding_version != "producer-job-v2":
            raise ContractValidationError(
                "reference_binding_version 必须为 producer-job-v2"
            )
        if value["verifier_version"] != "reference-v2":
            raise ContractValidationError(
                "producer-job-v2 证据必须使用 reference-v2 verifier"
            )
        value["reference_binding_version"] = binding_version
        value["jobs"] = _validate_investigation_evidence_records(
            value["jobs"],
            path="jobs",
            fields={
                "job_id",
                "terminal_status",
                "receipt_digest",
            },
            status_field="terminal_status",
            optional_fields={"submission_operation_id"},
        )
        value["artifacts"] = _validate_investigation_evidence_records(
            value["artifacts"],
            path="artifacts",
            fields={
                "artifact_id",
                "producer_job_id",
                "producer_submission_operation_id",
                "artifact_digest",
            },
            allow_empty=True,
        )
        for index, job in enumerate(value["jobs"]):
            if "submission_operation_id" in job:
                job["submission_operation_id"] = (
                    _validate_scope_submission_operation_id(
                        job["submission_operation_id"],
                        f"jobs[{index}].submission_operation_id",
                    )
                )
        for index, artifact in enumerate(value["artifacts"]):
            artifact["producer_submission_operation_id"] = (
                _validate_scope_submission_operation_id(
                    artifact["producer_submission_operation_id"],
                    f"artifacts[{index}].producer_submission_operation_id",
                )
            )
        jobs_by_id = {item["job_id"]: item for item in value["jobs"]}
        submission_operation_ids = [
            item["submission_operation_id"]
            for item in value["jobs"]
            if "submission_operation_id" in item
        ]
        if len(submission_operation_ids) != len(set(submission_operation_ids)):
            raise ContractValidationError(
                "producer-job-v2 证据中的 submission_operation_id 必须唯一"
            )
        for artifact in value["artifacts"]:
            producer = jobs_by_id.get(artifact["producer_job_id"])
            if producer is None:
                raise ContractValidationError(
                    "producer-job-v2 artifact 必须绑定已验证 producer job"
                )
            if (
                artifact["producer_submission_operation_id"]
                != producer.get("submission_operation_id")
            ):
                raise ContractValidationError(
                    "producer-job-v2 artifact 的提交 operation 与 producer job 不一致"
                )
    canonical_json(value)
    return dict(value)


def _validate_investigation_protocol_version(value: Any, path: str) -> int:
    """校验新 Investigation 事件使用的协议版本。"""

    if value != INVESTIGATION_PROTOCOL_VERSION:
        raise ContractValidationError(f"{path} 必须为 {INVESTIGATION_PROTOCOL_VERSION}")
    return value


def validate_task_review_session_payload(payload: Any) -> Dict[str, Any]:
    """校验每 task 唯一的 Review 会话冻结事实及其安全恢复身份。

    输入是 provider、模型、sandbox 策略摘要、workspace/CODEX_HOME、scope、thread
    和 operation 的安全身份；动作只做字段、路径、SHA-256 和 v8 协议检查，不访问
    provider 或读取 transcript；返回可追加的会话正文。缺字段、绝对路径、秘密正文、
    摘要格式错误或协议漂移时抛出合同错误，不产生副作用。
    """

    value = require_exact_keys(
        payload,
        {
            "task_id",
            "provider_id",
            "model",
            "sandbox_policy_digest",
            "workspace_ref",
            "codex_home_ref",
            "scope_id",
            "codex_thread_id",
            "operation_instance_id",
            "investigation_protocol_version",
        },
        "TaskReviewSessionStarted.payload",
    )
    for field in (
        "task_id",
        "provider_id",
        "model",
        "workspace_ref",
        "codex_home_ref",
        "scope_id",
        "codex_thread_id",
        "operation_instance_id",
    ):
        value[field] = _validate_investigation_identity(value[field], field)
    value["sandbox_policy_digest"] = _validate_investigation_digest(
        value["sandbox_policy_digest"], "sandbox_policy_digest"
    )
    value["investigation_protocol_version"] = _validate_investigation_protocol_version(
        value["investigation_protocol_version"], "investigation_protocol_version"
    )
    canonical_json(value)
    return dict(value)


def validate_task_review_turn_started_payload(payload: Any) -> Dict[str, Any]:
    """校验一次 Review turn 已获得 provider 接受的事实，不保存 transcript。

    输入是 task 级会话、批次和 provider turn 的安全身份；动作核对请求摘要、
    会话协议版本和从 1 递增的 turn_index；返回可追加到事件历史的正文，缺少会话
    引用、使用重复 turn 或携带 transcript/路径时抛出合同错误。
    """

    value = require_exact_keys(
        payload,
        {
            "task_id",
            "session_event_ref",
            "batch_id",
            "review_thread_id",
            "review_turn_id",
            "turn_index",
            "request_digest",
            "operation_instance_id",
            "investigation_protocol_version",
        },
        "TaskReviewTurnStarted.payload",
        optional_fields={
            "attempt_number",
            "attempt_status",
            "logical_operation_instance_id",
            "previous_attempt_closure_event_ref",
            "previous_attempt_terminal_event_ref",
        },
    )
    for field in (
        "task_id",
        "session_event_ref",
        "batch_id",
        "review_thread_id",
        "review_turn_id",
        "operation_instance_id",
    ):
        value[field] = _validate_investigation_identity(value[field], field)
    if (
        isinstance(value["turn_index"], bool)
        or not isinstance(value["turn_index"], int)
        or value["turn_index"] < 1
    ):
        raise ContractValidationError("turn_index 必须是从 1 开始的整数")
    value["request_digest"] = _validate_investigation_digest(
        value["request_digest"], "request_digest"
    )
    value["investigation_protocol_version"] = _validate_investigation_protocol_version(
        value["investigation_protocol_version"], "investigation_protocol_version"
    )
    _normalize_turn_attempt_fields(value, path="TaskReviewTurnStarted.payload")
    canonical_json(value)
    return dict(value)


def validate_investigation_repair_turn_started_payload(payload: Any) -> Dict[str, Any]:
    """校验 deepseek_repair turn 的成功启动事实及其 pending request 绑定。

    输入是 task、batch、组合、当前 request、1–3 的请求编号、session thread/scope、
    turn、请求摘要和 operation；动作只校验稳定身份、摘要和协议，不递增预算或写
    数据库；返回可追加的启动正文。编号跳跃、路径/秘密内容、协议漂移或字段不完整
    时抛出合同错误；repair_count 不由 Started 事件递增，只有引用该 turn 的 Result
    成功登记时才在同一事务更新当前索引。
    """

    value = require_exact_keys(
        payload,
        {
            "task_id",
            "batch_id",
            "combination_id",
            "request_event_ref",
            "repair_request_number",
            "codex_thread_id",
            "scope_id",
            "turn_id",
            "request_digest",
            "operation_instance_id",
            "investigation_protocol_version",
        },
        "InvestigationRepairTurnStarted.payload",
        optional_fields={
            "attempt_number",
            "attempt_status",
            "logical_operation_instance_id",
            "previous_attempt_closure_event_ref",
            "previous_attempt_terminal_event_ref",
        },
    )
    for field in (
        "task_id",
        "batch_id",
        "combination_id",
        "request_event_ref",
        "codex_thread_id",
        "scope_id",
        "turn_id",
        "operation_instance_id",
    ):
        value[field] = _validate_investigation_identity(value[field], field)
    if (
        isinstance(value["repair_request_number"], bool)
        or not isinstance(value["repair_request_number"], int)
        or not 1 <= value["repair_request_number"] <= 3
    ):
        raise ContractValidationError("repair_request_number 必须位于 1 到 3 之间")
    value["request_digest"] = _validate_investigation_digest(
        value["request_digest"], "request_digest"
    )
    value["investigation_protocol_version"] = _validate_investigation_protocol_version(
        value["investigation_protocol_version"], "investigation_protocol_version"
    )
    _normalize_turn_attempt_fields(value, path="InvestigationRepairTurnStarted.payload")
    canonical_json(value)
    return dict(value)


def validate_investigation_session_payload(payload: Any) -> Dict[str, Any]:
    """校验 Investigation 会话创建事实和其恢复身份。

    输入是已经入选的 batch/combination 及其独立 thread、workspace、scope 身份；
    动作只校验稳定引用，不读取外部工具或 transcript；返回可追加的会话正文，
    缺少身份、使用绝对路径或协议版本漂移时抛出合同错误。
    """

    value = require_exact_keys(
        payload,
        {
            "batch_id",
            "combination_id",
            "codex_thread_id",
            "workspace_ref",
            "scope_id",
            "operation_instance_id",
        },
        "InvestigationSessionStarted.payload",
        optional_fields={
            "task_id",
            "session_version",
            "session_id",
            "investigation_protocol_version",
        },
    )
    for field in (
        "batch_id",
        "combination_id",
        "workspace_ref",
        "scope_id",
        "operation_instance_id",
    ):
        value[field] = _validate_investigation_identity(value[field], field)
    if "task_id" in value:
        value["task_id"] = _validate_investigation_identity(value["task_id"], "task_id")
    if value.get("codex_thread_id") is None:
        if value.get("investigation_protocol_version") != INVESTIGATION_PROTOCOL_VERSION:
            raise ContractValidationError(
                "legacy Investigation session 必须在创建时绑定 codex_thread_id"
            )
    else:
        value["codex_thread_id"] = _validate_investigation_identity(
            value["codex_thread_id"], "codex_thread_id"
        )
    if "session_id" in value:
        value["session_id"] = _validate_investigation_identity(
            value["session_id"], "session_id"
        )
    if "session_version" in value:
        if (
            isinstance(value["session_version"], bool)
            or not isinstance(value["session_version"], int)
            or value["session_version"] != 1
        ):
            raise ContractValidationError("session_version 必须为 1")
    else:
        value["session_version"] = 1
    if "investigation_protocol_version" in value:
        value["investigation_protocol_version"] = _validate_investigation_protocol_version(
            value["investigation_protocol_version"], "investigation_protocol_version"
        )
    canonical_json(value)
    return dict(value)


def validate_investigation_session_thread_bound_payload(payload: Any) -> Dict[str, Any]:
    """校验把首次未知的 SDK 身份绑定到既有 Investigation session 的事实。

    输入是 task、batch、组合、稳定 session_id、thread、workspace、scope 和 operation
    身份；动作只校验 v8 协议和安全引用，不修改数据库或允许第二次绑定；返回可追加的
    ThreadBound 正文。未知 session、thread/身份漂移、路径/凭据或字段不完整时抛出
    合同错误。BatchCreated 预留的内部 workspace/scope 可在首次真实启动时被此事件
    一次性替换，事件本身记录最终绑定，之后不得再次改变。
    """

    value = require_exact_keys(
        payload,
        {
            "task_id",
            "batch_id",
            "combination_id",
            "session_id",
            "codex_thread_id",
            "workspace_ref",
            "scope_id",
            "operation_instance_id",
            "investigation_protocol_version",
        },
        "InvestigationSessionThreadBound.payload",
    )
    for field in (
        "task_id",
        "batch_id",
        "combination_id",
        "session_id",
        "codex_thread_id",
        "workspace_ref",
        "scope_id",
        "operation_instance_id",
    ):
        value[field] = _validate_investigation_identity(value[field], field)
    value["investigation_protocol_version"] = _validate_investigation_protocol_version(
        value["investigation_protocol_version"], "investigation_protocol_version"
    )
    canonical_json(value)
    return dict(value)


def validate_investigation_session_thread_migrated_payload(payload: Any) -> Dict[str, Any]:
    """校验 provider/auth 切换后的 Investigation thread 迁移事实。

    输入是同一 session 的旧 thread、新 thread、scope，以及 waiting sidecar 指定的
    job、operation、submission 和 turn checkpoint。动作只核对稳定身份并返回规范正文；
    旧 thread 不会被覆盖为当前值，因而事件历史仍可审计。缺字段、thread/scope/job
    identity 漂移或协议版本错误时抛出合同错误。
    """

    value = require_exact_keys(
        payload,
        {
            "task_id",
            "batch_id",
            "combination_id",
            "session_id",
            "source_codex_thread_id",
            "target_codex_thread_id",
            "workspace_ref",
            "scope_id",
            "active_job_id",
            "active_operation_id",
            "active_submission_key",
            "checkpoint_turn_id",
            "operation_instance_id",
            "investigation_protocol_version",
        },
        "InvestigationSessionThreadMigrated.payload",
    )
    for field in (
        "task_id",
        "batch_id",
        "combination_id",
        "session_id",
        "source_codex_thread_id",
        "target_codex_thread_id",
        "workspace_ref",
        "scope_id",
        "active_job_id",
        "active_operation_id",
        "active_submission_key",
        "checkpoint_turn_id",
        "operation_instance_id",
    ):
        value[field] = _validate_investigation_identity(value[field], field)
    if value["source_codex_thread_id"] == value["target_codex_thread_id"]:
        raise ContractValidationError("thread migration 的 source/target 不能相同")
    value["investigation_protocol_version"] = _validate_investigation_protocol_version(
        value["investigation_protocol_version"],
        "investigation_protocol_version",
    )
    canonical_json(value)
    return dict(value)


def validate_investigation_turn_started_payload(payload: Any) -> Dict[str, Any]:
    """校验首轮 Investigation turn 已成功进入 in-flight 的事实。

    输入是 session、batch、组合、已绑定 thread/scope、turn 和 operation 身份；动作只
    校验稳定引用与 v8 协议，不消费 repair 预算或保存 transcript；返回可追加正文。
    未知 session、thread/scope 漂移、重复身份、秘密内容或协议错误时抛出合同错误。
    """

    value = require_exact_keys(
        payload,
        {
            "task_id",
            "batch_id",
            "combination_id",
            "session_id",
            "codex_thread_id",
            "scope_id",
            "turn_id",
            "operation_instance_id",
            "investigation_protocol_version",
        },
        "InvestigationTurnStarted.payload",
        optional_fields={
            "attempt_number",
            "attempt_status",
            "logical_operation_instance_id",
            "previous_attempt_closure_event_ref",
            "previous_attempt_terminal_event_ref",
        },
    )
    for field in (
        "task_id",
        "batch_id",
        "combination_id",
        "session_id",
        "codex_thread_id",
        "scope_id",
        "turn_id",
        "operation_instance_id",
    ):
        value[field] = _validate_investigation_identity(value[field], field)
    value["investigation_protocol_version"] = _validate_investigation_protocol_version(
        value["investigation_protocol_version"], "investigation_protocol_version"
    )
    _normalize_turn_attempt_fields(value, path="InvestigationTurnStarted.payload")
    canonical_json(value)
    return dict(value)


def validate_investigation_turn_waiting_payload(payload: Any) -> Dict[str, Any]:
    """校验一次 Investigation/repair turn 已进入可恢复等待态的事实。

    输入是当前 initial 或 repair attempt 的完整 session/thread/turn/scope/workspace 摘要
    身份、logical 与 attempt operation、真实作业三元身份，以及分别表示 submit 参数
    和规范化 sidecar 记录的两个摘要；动作只校验稳定引用和 SHA-256，不访问作业服务
    或读取私有回执。该事实不能用于 Review，也不能省略任何恢复身份；字段缺失、身份
    漂移、摘要格式错误或协议版本错误时抛出合同错误，调用方事务不会写入半条等待事实。
    """

    # v9 早期 sidecar 使用 operation_id；读取旧事件时只在内存中改名，v10
    # 新事件始终只保存 submission_operation_id，避免把提交作业的操作身份与
    # 其它 operation 引用混在一起。
    if isinstance(payload, Mapping) and (
        "operation_id" in payload or "workspace_ref" in payload
    ):
        payload = dict(payload)
    if isinstance(payload, Mapping) and "operation_id" in payload:
        if "submission_operation_id" in payload:
            raise ContractValidationError(
                "InvestigationTurnWaiting 不能同时提供旧 operation_id 和 canonical submission_operation_id"
            )
        payload["submission_operation_id"] = payload.pop("operation_id")
    if isinstance(payload, Mapping) and "workspace_ref" in payload:
        if "workspace_ref_digest" in payload:
            raise ContractValidationError(
                "InvestigationTurnWaiting 不能同时提供旧 workspace_ref 和 canonical workspace_ref_digest"
            )
        workspace_ref = _validate_investigation_identity(
            payload.pop("workspace_ref"), "workspace_ref"
        )
        payload["workspace_ref_digest"] = hashlib.sha256(
            canonical_json(workspace_ref).encode("utf-8")
        ).hexdigest()
    value = require_exact_keys(
        payload,
        {
            "task_id",
            "turn_started_event_ref",
            "kind",
            "batch_id",
            "combination_id",
            "session_id",
            "codex_thread_id",
            "workspace_ref_digest",
            "scope_id",
            "turn_id",
            "attempt_number",
            "logical_operation_instance_id",
            "attempt_operation_instance_id",
            "job_id",
            "submission_operation_id",
            "submission_key",
            "submission_request_digest",
            "sidecar_record_digest",
            "investigation_protocol_version",
        },
        "InvestigationTurnWaiting.payload",
    )
    for field in (
        "task_id",
        "turn_started_event_ref",
        "batch_id",
        "combination_id",
        "session_id",
        "codex_thread_id",
        "scope_id",
        "turn_id",
        "logical_operation_instance_id",
        "attempt_operation_instance_id",
        "job_id",
        "submission_operation_id",
        "submission_key",
    ):
        value[field] = _validate_investigation_identity(value[field], field)
    if value["kind"] not in {"initial", "repair"}:
        raise ContractValidationError(
            "InvestigationTurnWaiting.kind 只能是 initial 或 repair"
        )
    if (
        isinstance(value["attempt_number"], bool)
        or not isinstance(value["attempt_number"], int)
        or value["attempt_number"] < 1
    ):
        raise ContractValidationError("InvestigationTurnWaiting.attempt_number 必须是正整数")
    for field in (
        "workspace_ref_digest",
        "submission_request_digest",
        "sidecar_record_digest",
    ):
        value[field] = _validate_investigation_digest(value[field], field)
    value["investigation_protocol_version"] = _validate_investigation_protocol_version(
        value["investigation_protocol_version"],
        "investigation_protocol_version",
    )
    canonical_json(value)
    return dict(value)


def _validate_investigation_terminal_marker_refs(value: Dict[str, Any]) -> None:
    """校验 TaskClosed 使用的五项 marker 来源引用。

    输入是已确认字段集合的终态正文；动作要求来源是排序且去重的字符串数组，并在
    内存中规范化每个稳定引用。返回值为空，来源格式不符时抛出合同错误，不写入状态。
    """

    marker_refs = value["marker_event_refs"]
    if (
        not isinstance(marker_refs, list)
        or any(not isinstance(item, str) for item in marker_refs)
        or marker_refs != sorted(set(marker_refs))
    ):
        raise ContractValidationError("marker_event_refs 必须是排序且去重的字符串数组")
    value["marker_event_refs"] = [
        _validate_investigation_reference(item, f"marker_event_refs[{index}]")
        for index, item in enumerate(marker_refs)
    ]


def _validate_investigation_terminal_answer_fields(value: Dict[str, Any]) -> None:
    """校验终态回答布尔值和可选正文。

    输入是终态正文；动作检查 accepted/can_answer 类型并规范化回答与不可回答说明，
    返回值为空，类型或私有内容错误时抛出合同错误，不产生数据库副作用。
    """

    if not isinstance(value["accepted"], bool) or not isinstance(value["can_answer"], bool):
        raise ContractValidationError("accepted 和 can_answer 必须是 boolean")
    for field in ("final_answer", "answerability_explanation"):
        if value[field] is not None:
            value[field] = _validate_investigation_text(value[field], field)


def _validate_investigation_accepted_terminal_fields(value: Mapping[str, Any]) -> None:
    """校验 accepted 终态的回答字段。

    输入是已通过类型校验的终态正文；动作要求 accepted 同时可回答且有最终答案，
    返回值为空，回答字段矛盾时抛出合同错误。accepted 可以同时携带部分
    ``revise_hypothesis`` marker：这些 marker 表示同一批次中已经被证伪的组合，
    不会把整批结果改成假设重生。
    """

    if not value["can_answer"] or value["final_answer"] is None:
        raise ContractValidationError("accepted 终态必须 can_answer 且有 final_answer")
    if value["answerability_explanation"] is not None:
        raise ContractValidationError("accepted 终态不能同时保存 answerability_explanation")


def _validate_investigation_unaccepted_terminal_fields(value: Mapping[str, Any]) -> None:
    """校验非 accepted 终态的不可回答说明或 marker 来源。

    输入是已通过类型校验的终态正文；动作拒绝可回答内容，并要求至少保留说明或
    marker 来源，返回值为空，字段矛盾时抛出合同错误。
    """

    if value["can_answer"] or value["final_answer"] is not None:
        raise ContractValidationError("非 accepted 终态不能保存可回答内容")
    if value["answerability_explanation"] is None and not value["marker_event_refs"]:
        raise ContractValidationError("没有 marker 的非 accepted 终态必须说明不可回答原因")


def _validate_investigation_terminal_outcome_fields(value: Mapping[str, Any]) -> None:
    """校验 accepted 与非 accepted 终态的互斥关系。

    输入是已完成基础字段规范化的终态正文；动作按 accepted 分派对应规则，返回值为
    空，任一终态逻辑不一致时抛出合同错误。
    """

    if value["accepted"]:
        _validate_investigation_accepted_terminal_fields(value)
    else:
        _validate_investigation_unaccepted_terminal_fields(value)


def validate_investigation_task_terminal_payload(payload: Any) -> Dict[str, Any]:
    """校验 Investigation v2 任务终态及其 Review 来源。

    输入是批次、accepted/answerability、回答说明、当前 Review 和五项 marker 来源；
    动作检查终态字段之间的一致性和 v8 协议，不修改任务状态；返回可追加的 TaskClosed
    正文。回答缺失、来源重复、失败条件矛盾或字段含私有内容时抛出合同错误。
    """

    value = require_exact_keys(
        payload,
        {
            "task_id",
            "batch_id",
            "accepted",
            "can_answer",
            "final_answer",
            "answerability_explanation",
            "review_event_ref",
            "marker_event_refs",
            "operation_instance_id",
            "investigation_protocol_version",
        },
        "InvestigationTaskClosed.payload",
    )
    for field in (
        "task_id",
        "batch_id",
        "review_event_ref",
        "operation_instance_id",
    ):
        value[field] = _validate_investigation_identity(value[field], field)
    _validate_investigation_terminal_marker_refs(value)
    _validate_investigation_terminal_answer_fields(value)
    _validate_investigation_terminal_outcome_fields(value)
    value["investigation_protocol_version"] = _validate_investigation_protocol_version(
        value["investigation_protocol_version"], "investigation_protocol_version"
    )
    canonical_json(value)
    return dict(value)


def validate_hypothesis_selection_completed_payload(payload: Any) -> Dict[str, Any]:
    """校验只完成 Top5 筛选的任务终态事实。

    输入是已提交的 Hypothesis batch、五个入选候选和 finalized manifest 摘要；动作
    只核对选择身份、排名和来源，不创建 Investigation session，也不判断科学答案。
    返回值是可追加到 Working Memory 的规范正文；数量、排名、摘要或协议不一致时
    抛出合同错误。
    """

    value = require_exact_keys(
        payload,
        {
            "task_id",
            "batch_id",
            "batch_index",
            "selected_count",
            "selected_combinations",
            "manifest_sha256",
            "operation_instance_id",
            "investigation_protocol_version",
        },
        "HypothesisSelectionCompleted.payload",
    )
    for field in ("task_id", "batch_id", "operation_instance_id"):
        value[field] = _validate_investigation_identity(value[field], field)
    for field in ("batch_index", "selected_count"):
        value[field] = require_json_integer(value[field], field, 1)
    if value["selected_count"] != 5:
        raise ContractValidationError(
            "HypothesisSelectionCompleted.selected_count 必须固定为 5"
        )
    selected = value["selected_combinations"]
    if not isinstance(selected, list) or len(selected) != 5:
        raise ContractValidationError(
            "HypothesisSelectionCompleted.selected_combinations 必须恰有五项"
        )
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(selected, start=1):
        if not isinstance(item, Mapping) or set(item) != {
            "combination_id",
            "combination_index",
            "hypothesis_id",
            "rank",
        }:
            raise ContractValidationError(
                f"selected_combinations[{index - 1}] 字段不完整"
            )
        normalized_item = {
            "combination_id": _validate_investigation_identity(
                item["combination_id"],
                f"selected_combinations[{index - 1}].combination_id",
            ),
            "combination_index": require_json_integer(
                item["combination_index"],
                f"selected_combinations[{index - 1}].combination_index",
                1,
            ),
            "hypothesis_id": _validate_investigation_identity(
                item["hypothesis_id"],
                f"selected_combinations[{index - 1}].hypothesis_id",
            ),
            "rank": require_json_integer(
                item["rank"],
                f"selected_combinations[{index - 1}].rank",
                1,
            ),
        }
        normalized.append(normalized_item)
    if [item["combination_index"] for item in normalized] != [1, 2, 3, 4, 5]:
        raise ContractValidationError(
            "HypothesisSelectionCompleted.combination_index 必须覆盖 1 到 5"
        )
    if [item["rank"] for item in normalized] != [1, 2, 3, 4, 5]:
        raise ContractValidationError(
            "HypothesisSelectionCompleted.rank 必须覆盖 1 到 5"
        )
    if len({item["combination_id"] for item in normalized}) != 5:
        raise ContractValidationError("HypothesisSelectionCompleted 的 combination_id 必须唯一")
    if len({item["hypothesis_id"] for item in normalized}) != 5:
        raise ContractValidationError("HypothesisSelectionCompleted 的 hypothesis_id 必须唯一")
    value["selected_combinations"] = normalized
    if not isinstance(value["manifest_sha256"], str) or not _SHA256_RE.fullmatch(
        value["manifest_sha256"]
    ):
        raise ContractValidationError(
            "HypothesisSelectionCompleted.manifest_sha256 必须是 64 位小写 SHA-256"
        )
    value["investigation_protocol_version"] = _validate_investigation_protocol_version(
        value["investigation_protocol_version"],
        "investigation_protocol_version",
    )
    canonical_json(value)
    return dict(value)


def validate_investigation_task_failure_payload(payload: Any) -> Dict[str, Any]:
    """校验协调器或 Controller 已确认的 Investigation 执行失败事实。

    输入是任务身份、固定失败原因和异常类型；动作只接受不含路径、凭据或异常正文
    的安全标识，返回可追加到 Working Memory 的失败摘要。它不判断科学结论，也不
    把等待作业或可恢复暂停转换成失败；缺字段、标识不安全或正文夹带私有信息时
    抛出合同错误且不修改数据库。
    """

    value = require_exact_keys(
        payload,
        {"task_id", "reason_code", "error_type"},
        "InvestigationTaskFailed.payload",
    )
    value["task_id"] = _validate_investigation_identity(value["task_id"], "task_id")
    for field in ("reason_code", "error_type"):
        candidate = value[field]
        if not isinstance(candidate, str) or not re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_.-]{0,127}", candidate
        ):
            raise ContractValidationError(
                f"{field} 必须是安全的 ASCII 标识"
            )
        value[field] = candidate
    canonical_json(value)
    return dict(value)


def validate_investigation_task_correction_payload(payload: Any) -> Dict[str, Any]:
    """校验维护者显式提交的 Investigation 任务更正事实。

    输入必须明确绑定任务、当前 TaskClosed、当前 Review、当前 Result、组合和修复
    建议；可选的 batch、结果版本、operation 和协议字段在服务边界补齐后也会严格
    校验。动作只规范化安全身份和公开建议，不判断科学内容、不访问数据库；缺少任一
    必需绑定、身份不安全、建议含私有路径或协议漂移时抛出合同错误。
    """

    value = require_exact_keys(
        payload,
        {
            "task_id",
            "terminal_event_ref",
            "review_event_ref",
            "result_event_ref",
            "combination_id",
            "repair_advice",
        },
        "InvestigationTaskCorrection.payload",
        optional_fields={
            "batch_id",
            "result_version",
            "operation_instance_id",
            "investigation_protocol_version",
        },
    )
    for field in (
        "task_id",
        "terminal_event_ref",
        "review_event_ref",
        "result_event_ref",
        "combination_id",
        "batch_id",
        "result_version",
        "operation_instance_id",
    ):
        if field in value:
            value[field] = _validate_investigation_identity(value[field], field)
    value["repair_advice"] = _validate_investigation_text(
        value["repair_advice"], "repair_advice"
    )
    if "investigation_protocol_version" in value:
        value["investigation_protocol_version"] = _validate_investigation_protocol_version(
            value["investigation_protocol_version"],
            "investigation_protocol_version",
        )
    canonical_json(value)
    return dict(value)


def _validate_investigation_result_identity_fields(value: Dict[str, Any]) -> None:
    """校验 InvestigationResult 的组合、线程和可选 task 身份。

    输入是已完成精确字段检查的结果正文；动作规范化稳定身份，返回值为空，身份
    类型或路径不安全时抛出合同错误，不访问外部服务。
    """

    for field in (
        "batch_id",
        "combination_id",
        "result_version",
        "codex_thread_id",
        "scope_id",
        "turn_id",
    ):
        value[field] = _validate_investigation_identity(value[field], field)
    if "task_id" in value:
        value["task_id"] = _validate_investigation_identity(value["task_id"], "task_id")


def _validate_investigation_result_claim_fields(value: Dict[str, Any]) -> None:
    """校验 Result 的自判定和叙述正文。

    输入是结果正文；动作只接受 supported/refuted 并过滤私有内容，返回值为空，
    枚举或正文不合法时抛出合同错误。
    """

    if not isinstance(value["self_judgment"], str) or value["self_judgment"] not in {
        "supported",
        "refuted",
    }:
        raise ContractValidationError("self_judgment 必须是 supported 或 refuted")
    value["narrative"] = _validate_investigation_text(value["narrative"], "narrative")


def _validate_investigation_result_reference_fields(value: Dict[str, Any]) -> None:
    """校验 Result 的 job、operation 和 artifact 来源引用。

    输入是结果正文；动作在内存中规范化三个稳定引用数组，返回值为空，空值、重复
    引用或本机路径时抛出合同错误。
    """

    for field in ("job_refs", "operation_refs", "artifact_refs"):
        value[field] = _validate_investigation_reference_list(value[field], field)


def _validate_investigation_result_text_list(
    value: Dict[str, Any],
    field: str,
) -> None:
    """校验 Result 的 observations 或 limitations 正文数组。

    输入是结果正文和字段名；动作逐项过滤正文并拒绝 observations 空数组或重复项，
    返回值为空，非字符串正文和私有内容时抛出合同错误。
    """

    raw_items = value[field] if isinstance(value[field], (list, tuple)) else ()
    value[field] = [
        _validate_investigation_text(item, f"{field}[{index}]")
        for index, item in enumerate(raw_items)
    ]
    if not isinstance(value[field], list) or (
        field == "observations" and not value[field]
    ):
        raise ContractValidationError(
            f"{field} 必须是完整正文数组"
            + ("且至少包含一项" if field == "observations" else "")
        )
    if len(value[field]) != len(set(value[field])):
        raise ContractValidationError(f"{field} 不能包含重复正文")


def _validate_investigation_result_text_fields(value: Dict[str, Any]) -> None:
    """校验 Result 的 observations 和 limitations 两组正文。

    输入是结果正文；动作复用单字段正文校验并写回规范化数组，返回值为空，正文
    不完整或重复时抛出合同错误。
    """

    for field in ("observations", "limitations"):
        _validate_investigation_result_text_list(value, field)


def _validate_investigation_result_optional_reference_fields(
    value: Dict[str, Any],
) -> None:
    """校验 Result 的 operation、session、evidence 和 repair 来源引用。

    输入是结果正文；动作只规范化存在的可选稳定引用，返回值为空，引用不安全时
    抛出合同错误，不访问数据库或 provider。
    """

    for field in (
        "operation_instance_id",
        "session_event_ref",
        "evidence_verification_event_ref",
        "repair_turn_event_ref",
    ):
        if field in value:
            value[field] = _validate_investigation_reference(value[field], field)


def _validate_investigation_result_protocol_field(value: Dict[str, Any]) -> None:
    """校验 Result 若声明的 Investigation 协议版本。

    输入是结果正文；动作在存在协议字段时固定校验为 v8，返回值为空，协议漂移时
    抛出合同错误。
    """

    if "investigation_protocol_version" in value:
        value["investigation_protocol_version"] = _validate_investigation_protocol_version(
            value["investigation_protocol_version"], "investigation_protocol_version"
        )


def validate_investigation_result_payload(payload: Any) -> Dict[str, Any]:
    """校验一版完整 InvestigationResult 科学事实。

    输入是单个组合的一版叙述、观测、限制及安全来源摘要；动作拒绝私有 transcript、
    token 和绝对路径，并在 v8 正文中要求上层已验证证据事件及返修 turn；返回规范化
    结果正文，身份漂移、引用缺失或版本不合法时抛出合同错误。
    """

    value = require_exact_keys(
        payload,
        {
            "batch_id",
            "combination_id",
            "result_version",
            "codex_thread_id",
            "scope_id",
            "turn_id",
            "self_judgment",
            "narrative",
            "job_refs",
            "operation_refs",
            "artifact_refs",
            "observations",
            "limitations",
        },
        "InvestigationResultRecorded.payload",
        optional_fields={
            "task_id",
            "operation_instance_id",
            "session_event_ref",
            "evidence_verification_event_ref",
            "repair_turn_event_ref",
            "investigation_protocol_version",
        },
    )
    _validate_investigation_result_identity_fields(value)
    _validate_investigation_result_claim_fields(value)
    _validate_investigation_result_reference_fields(value)
    _validate_investigation_result_text_fields(value)
    _validate_investigation_result_optional_reference_fields(value)
    _validate_investigation_result_protocol_field(value)
    canonical_json(value)
    return dict(value)


def _normalize_result_version_vector(value: Any, path: str) -> List[List[str]]:
    """把一项或五项结果版本向量规范成排序后的二元数组。"""

    if isinstance(value, Mapping):
        pairs = list(value.items())
    elif isinstance(value, (list, tuple)):
        pairs = list(value)
    else:
        raise ContractValidationError(f"{path} 必须是一项或五项组合与版本配对")
    normalized: list[list[str]] = []
    for index, pair in enumerate(pairs):
        if isinstance(pair, Mapping):
            if set(pair) != {"combination_id", "result_version"}:
                raise ContractValidationError(f"{path}[{index}] 字段不完整")
            combination_id, result_version = pair["combination_id"], pair["result_version"]
        elif isinstance(pair, (list, tuple)) and len(pair) == 2:
            combination_id, result_version = pair
        else:
            raise ContractValidationError(f"{path}[{index}] 必须是二元配对")
        normalized.append(
            [
                _validate_investigation_identity(combination_id, f"{path}[{index}].combination_id"),
                _validate_investigation_identity(result_version, f"{path}[{index}].result_version"),
            ]
        )
    if len(normalized) not in {1, 5} or len({pair[0] for pair in normalized}) != len(normalized):
        raise ContractValidationError(f"{path} 必须完整包含一个或五个唯一组合")
    return sorted(normalized)


def validate_investigation_result_version_vector(
    value: Any,
    path: str = "result_version_vector",
) -> List[List[str]]:
    """校验并规范完整的一项或五项 Investigation Result 版本向量。

    输入是组合身份与结果版本的 mapping 或二元数组；动作要求恰有一个或五个唯一组合，
    返回按组合身份排序的副本，不写入数据库。缺项、重复、非字符串身份或恶意
    非对象输入统一抛出合同错误，供 Review replay 读取和写入合同复用。
    """

    return _normalize_result_version_vector(value, path)


def _validate_investigation_review_identity_fields(value: Dict[str, Any]) -> None:
    """校验批次 Review 的 task、线程、turn 与 operation 稳定引用。

    输入是已完成字段集合检查的 Review 正文；动作规范化稳定引用、事件引用和协议
    版本，不写入数据库；身份、路径或协议错误时抛出合同错误。
    """

    for field in ("batch_id", "review_thread_id", "review_turn_id"):
        value[field] = _validate_investigation_identity(value[field], field)
    if "task_id" in value:
        value["task_id"] = _validate_investigation_identity(value["task_id"], "task_id")
    if "operation_instance_id" in value:
        value["operation_instance_id"] = _validate_investigation_reference(
            value["operation_instance_id"], "operation_instance_id"
        )
    for field in ("task_review_session_event_ref", "task_review_turn_event_ref"):
        if field in value:
            value[field] = _validate_investigation_reference(value[field], field)
    if "investigation_protocol_version" in value:
        value["investigation_protocol_version"] = _validate_investigation_protocol_version(
            value["investigation_protocol_version"], "investigation_protocol_version"
        )


def _normalize_investigation_review_version_fields(
    value: Dict[str, Any],
) -> set[str]:
    """校验并规范 Review 的一项或五项组合/版本向量，返回组合身份集合。

    输入是已完成字段集合检查的 Review 正文；动作校验当前批次组合和版本、可选的显式
    result_version_vector，并把三种表示统一为排序后的向量；字段缺失、重复或漂移
    时抛出合同错误。
    """

    combination_ids = (
        [
            _validate_investigation_identity(item, f"combination_ids[{index}]")
            for index, item in enumerate(value["combination_ids"])
        ]
        if isinstance(value["combination_ids"], (list, tuple))
        else []
    )
    versions = (
        [
            _validate_investigation_identity(item, f"result_versions[{index}]")
            for index, item in enumerate(value["result_versions"])
        ]
        if isinstance(value["result_versions"], (list, tuple))
        else []
    )
    if (
        len(combination_ids) not in {1, 5}
        or len(versions) != len(combination_ids)
        or len(set(combination_ids)) != len(combination_ids)
    ):
        raise ContractValidationError("Review 必须绑定完整的一项或五项结果版本向量")
    expected_vector = [list(item) for item in sorted(zip(combination_ids, versions))]
    if "result_version_vector" in value:
        vector = _normalize_result_version_vector(
            value["result_version_vector"], "result_version_vector"
        )
        if vector != expected_vector:
            raise ContractValidationError("result_version_vector 与组合/版本字段漂移")
    value["combination_ids"] = [item[0] for item in expected_vector]
    value["result_versions"] = [item[1] for item in expected_vector]
    value["result_version_vector"] = expected_vector
    return set(value["combination_ids"])


def _validate_investigation_review_finding(
    raw: Any,
    index: int,
) -> Dict[str, Any]:
    """校验单个 Review finding 的 pass/fail 互斥字段。

    输入是一个 finding 和其数组位置；动作核对组合身份、判断枚举与 repair advice，
    返回规范化副本，不写入状态；字段矛盾或私有正文时抛出合同错误。
    """

    finding = require_exact_keys(
        raw,
        {"combination_id", "validity", "hypothesis_judgment", "repair_advice"},
        f"findings[{index}]",
    )
    finding = dict(finding)
    finding["combination_id"] = _validate_investigation_identity(
        finding["combination_id"], f"findings[{index}].combination_id"
    )
    if not isinstance(finding["validity"], str) or finding["validity"] not in {
        "pass",
        "fail",
    }:
        raise ContractValidationError("finding.validity 必须是 pass 或 fail")
    if finding["validity"] == "pass":
        if not isinstance(finding["hypothesis_judgment"], str) or finding[
            "hypothesis_judgment"
        ] not in {"supported", "refuted"}:
            raise ContractValidationError("validity=pass 必须有 supported/refuted")
        if finding["repair_advice"] is not None:
            raise ContractValidationError("validity=pass 不能有 repair_advice")
    else:
        if finding["hypothesis_judgment"] is not None:
            raise ContractValidationError("validity=fail 不能有 hypothesis_judgment")
        finding["repair_advice"] = _validate_investigation_text(
            finding["repair_advice"], f"findings[{index}].repair_advice"
        )
    return finding


def _validate_investigation_review_findings(
    value: Dict[str, Any],
    combination_ids: set[str],
) -> None:
    """校验当前批次 finding 并按组合身份排序写回 Review 正文。

    输入是 Review 正文和完整组合身份集合；动作逐项校验、核对覆盖范围并写回排序后
    的内存字典，不产生数据库副作用；缺项、重复或跨组合 finding 时抛出合同错误。
    """

    findings = value["findings"]
    if not isinstance(findings, (list, tuple)) or len(findings) not in {1, 5}:
        raise ContractValidationError("Review 必须包含一项或五项 finding")
    normalized_findings = [
        _validate_investigation_review_finding(raw, index)
        for index, raw in enumerate(findings)
    ]
    if {item["combination_id"] for item in normalized_findings} != combination_ids:
        raise ContractValidationError("findings 没有完整覆盖五个组合")
    value["findings"] = sorted(
        normalized_findings,
        key=lambda item: item["combination_id"],
    )


def _read_investigation_review_answerability_fields(
    value: Mapping[str, Any],
) -> tuple[bool, str | None, str | None, bool, bool]:
    """读取并校验 Review 的回答字段，返回一致性判断所需的五个值。

    输入是已完成 finding 校验的 Review 正文；动作校验回答文本和 finding 汇总，不写
    入数据库；类型或私有正文错误时抛出合同错误，返回五元判断值。
    """

    if not isinstance(value["can_answer"], bool):
        raise ContractValidationError("can_answer 必须是 boolean")
    final_answer = value.get("final_answer")
    explanation = value.get("answerability_explanation")
    if final_answer is not None:
        final_answer = _validate_investigation_text(final_answer, "final_answer")
    if explanation is not None:
        explanation = _validate_investigation_text(explanation, "answerability_explanation")
    has_fail = any(item["validity"] == "fail" for item in value["findings"])
    has_supported = any(
        item["validity"] == "pass" and item["hypothesis_judgment"] == "supported"
        for item in value["findings"]
    )
    return value["can_answer"], final_answer, explanation, has_fail, has_supported


def _validate_investigation_review_answerability_consistency(
    can_answer: bool,
    final_answer: str | None,
    explanation: str | None,
    has_fail: bool,
    has_supported: bool,
) -> None:
    """校验五项 finding 与回答字段的逻辑一致性。

    输入是回答能力、最终回答、说明及 finding 汇总布尔值；动作只检查互斥关系，不
    修改状态；supported/fail 与回答字段矛盾时抛出合同错误。
    """

    if can_answer:
        if has_fail or not has_supported or final_answer is None or explanation is not None:
            raise ContractValidationError("can_answer=true 的五项判断或回答字段不一致")
    elif has_fail or not has_supported:
        if final_answer is not None or explanation is not None:
            raise ContractValidationError("fail 或全部 refuted 时不能保存回答字段")
    elif explanation is None or final_answer is not None:
        raise ContractValidationError("有 supported 但 can_answer=false 时必须保存 explanation")


def _validate_investigation_review_answerability(value: Dict[str, Any]) -> None:
    """校验 Review 的回答能力、最终回答与不可回答说明之间的一致性。

    输入是已规范化 finding 的 Review 正文；动作读取并核对回答字段后写回规范化的
    final_answer/explanation，仅修改内存字典；类型或逻辑矛盾时抛出合同错误。
    """

    answerability = _read_investigation_review_answerability_fields(value)
    _validate_investigation_review_answerability_consistency(*answerability)
    _can_answer, final_answer, explanation, _has_fail, _has_supported = answerability
    value["final_answer"] = final_answer
    value["answerability_explanation"] = explanation


def validate_investigation_batch_review_payload(payload: Any) -> Dict[str, Any]:
    """校验整批 Review 的五项版本向量、逐项判断与回答能力。

    输入是五个当前结果的完整版本向量、逐组合 finding 和回答说明；动作检查
    supported/refuted、fail advice、task 级 Review 会话引用及 vector 一致性；返回
    可追加正文，缺一项结果、旧版本、过期会话或回答字段矛盾时抛出合同错误。
    """

    value = require_exact_keys(
        payload,
        {
            "batch_id",
            "combination_ids",
            "result_versions",
            "findings",
            "can_answer",
            "review_thread_id",
            "review_turn_id",
        },
        "InvestigationBatchReviewed.payload",
        optional_fields={
            "task_id",
            "final_answer",
            "answerability_explanation",
            "operation_instance_id",
            "result_version_vector",
            "task_review_session_event_ref",
            "task_review_turn_event_ref",
            "investigation_protocol_version",
        },
    )
    _validate_investigation_review_identity_fields(value)
    combination_ids = _normalize_investigation_review_version_fields(value)
    _validate_investigation_review_findings(value, combination_ids)
    _validate_investigation_review_answerability(value)
    canonical_json(value)
    return dict(value)


def validate_investigation_repair_requested_payload(payload: Any) -> Dict[str, Any]:
    """校验一次由 Review 触发的返修请求，并限制其请求编号范围。

    输入是 Review 的 fail finding、当前 result 版本和请求编号；动作只保存建议文本
    与事实引用，不提前消耗 repair 预算；返回请求正文，缺失 Review/结果引用或编号
    越界时抛出合同错误，真正的预算递增留给 repair turn started 事件。
    """

    value = require_exact_keys(
        payload,
        {
            "batch_id",
            "combination_id",
            "result_version",
            "repair_count",
            "repair_advice",
            "review_event_ref",
            "operation_instance_id",
        },
        "InvestigationRepairRequested.payload",
        optional_fields={
            "task_id",
            "repair_request_number",
            "investigation_protocol_version",
        },
    )
    for field in ("batch_id", "combination_id", "result_version", "review_event_ref", "operation_instance_id"):
        value[field] = _validate_investigation_reference(value[field], field)
    if "task_id" in value:
        value["task_id"] = _validate_investigation_identity(value["task_id"], "task_id")
    if (
        isinstance(value["repair_count"], bool)
        or not isinstance(value["repair_count"], int)
        or not 1 <= value["repair_count"] <= 3
    ):
        raise ContractValidationError("repair_count 必须位于 1 到 3 之间")
    value["repair_advice"] = _validate_investigation_text(
        value["repair_advice"], "repair_advice"
    )
    if "repair_request_number" in value:
        if value["repair_request_number"] != value["repair_count"]:
            raise ContractValidationError("repair_request_number 与 repair_count 不一致")
    if "investigation_protocol_version" in value:
        value["investigation_protocol_version"] = _validate_investigation_protocol_version(
            value["investigation_protocol_version"], "investigation_protocol_version"
        )
    canonical_json(value)
    return dict(value)


def validate_investigation_repair_completed_payload(payload: Any) -> Dict[str, Any]:
    """校验返修完成事实，并要求它引用请求和新结果。

    输入是当前 pending request、成功生成的新 result 和对应 repair turn；动作只
    校验安全身份，持久层再核对结果事件确实因果引用 request/turn；返回完成正文，
    旧结果、异批结果或缺失因果来源时抛出合同错误。
    """

    value = require_exact_keys(
        payload,
        {
            "batch_id",
            "combination_id",
            "repair_count",
            "request_event_ref",
            "result_event_ref",
            "operation_instance_id",
        },
        "InvestigationRepairCompleted.payload",
        optional_fields={
            "task_id",
            "result_version",
            "repair_turn_event_ref",
            "investigation_protocol_version",
        },
    )
    for field in (
        "batch_id",
        "combination_id",
        "request_event_ref",
        "result_event_ref",
        "operation_instance_id",
    ):
        value[field] = _validate_investigation_reference(value[field], field)
    if "task_id" in value:
        value["task_id"] = _validate_investigation_identity(value["task_id"], "task_id")
    if "result_version" in value:
        value["result_version"] = _validate_investigation_identity(
            value["result_version"], "result_version"
        )
    if "repair_turn_event_ref" in value:
        value["repair_turn_event_ref"] = _validate_investigation_reference(
            value["repair_turn_event_ref"], "repair_turn_event_ref"
        )
    if (
        isinstance(value["repair_count"], bool)
        or not isinstance(value["repair_count"], int)
        or not 1 <= value["repair_count"] <= 3
    ):
        raise ContractValidationError("repair_count 必须位于 1 到 3 之间")
    if "investigation_protocol_version" in value:
        value["investigation_protocol_version"] = _validate_investigation_protocol_version(
            value["investigation_protocol_version"], "investigation_protocol_version"
        )
    canonical_json(value)
    return dict(value)


def validate_investigation_route_marker_payload(payload: Any) -> Dict[str, Any]:
    """校验不承载科学原因、只引用事实来源的 revise_hypothesis marker。

    输入是组合身份和真实 Result/Review 事件引用；动作不接受科学理由、自由文本或
    伪造 gate 字段；返回只含路由标记与来源的正文，来源缺失、版本漂移或协议非法时
    抛出合同错误，marker 本身不会替代科学 Review。
    """

    value = require_exact_keys(
        payload,
        {"batch_id", "combination_id", "marker"},
        "InvestigationRouteMarked.payload",
        optional_fields={
            "task_id",
            "result_event_ref",
            "review_event_ref",
            "investigation_protocol_version",
        },
    )
    for field in ("batch_id", "combination_id"):
        value[field] = _validate_investigation_identity(value[field], field)
    if "task_id" in value:
        value["task_id"] = _validate_investigation_identity(value["task_id"], "task_id")
    if value["marker"] != "revise_hypothesis":
        raise ContractValidationError("route marker 只能是 revise_hypothesis")
    for field in ("result_event_ref", "review_event_ref"):
        if field in value:
            value[field] = _validate_investigation_reference(value[field], field)
    if "investigation_protocol_version" in value:
        value["investigation_protocol_version"] = _validate_investigation_protocol_version(
            value["investigation_protocol_version"], "investigation_protocol_version"
        )
    if not {"result_event_ref", "review_event_ref"} <= set(value):
        raise ContractValidationError("route marker 必须同时引用真实 result 和 review 事实")
    canonical_json(value)
    return dict(value)


# 兼容阶段装配代码中更短的公开命名；这些包装器明确指向 canonical 合同，避免读者
# 把短名称误认为另一套事实格式。
def validate_investigation_started_payload(payload: Any) -> Dict[str, Any]:
    """兼容旧装配器，按 canonical SessionStarted 合同校验输入。"""

    return validate_investigation_session_payload(payload)


def validate_batch_review_payload(payload: Any) -> Dict[str, Any]:
    """兼容旧装配器，按 canonical InvestigationBatchReviewed 合同校验输入。"""

    return validate_investigation_batch_review_payload(payload)


def validate_repair_requested_payload(payload: Any) -> Dict[str, Any]:
    """兼容旧装配器，按 canonical InvestigationRepairRequested 合同校验输入。"""

    return validate_investigation_repair_requested_payload(payload)


def validate_repair_completed_payload(payload: Any) -> Dict[str, Any]:
    """兼容旧装配器，按 canonical InvestigationRepairCompleted 合同校验输入。"""

    return validate_investigation_repair_completed_payload(payload)


def validate_revise_hypothesis_route_payload(payload: Any) -> Dict[str, Any]:
    """兼容旧装配器，按 canonical InvestigationRouteMarked 合同校验输入。"""

    return validate_investigation_route_marker_payload(payload)


def _read_task_created_payload_mapping(payload: Any) -> Dict[str, Any]:
    """检查 TaskCreated 顶层字段并返回浅副本。"""

    if not isinstance(payload, dict):
        raise ContractValidationError("TaskCreated.payload 必须是 JSON object")
    required_keys = {
        "scientific_question",
        "chemistry_context",
        "task_config",
        "task_output_relative_dir",
        "tool_catalog_hash",
    }
    allowed_keys = required_keys | {
        "model_configuration",
        "expert_backend_selection",
        "case_profile",
        "gaussian_route_policy",
        "stage_architecture_version",
        "hypothesis_import",
    }
    missing = sorted(required_keys - set(payload))
    extra = sorted(set(payload) - allowed_keys)
    if missing or extra:
        raise ContractValidationError(
            f"TaskCreated.payload 字段不匹配：missing={missing}, extra={extra}"
        )
    return dict(payload)


def _validate_task_created_hypothesis_import(value: Dict[str, Any]) -> None:
    """校验可恢复的作者假设导入模式元数据。

    输入是 TaskCreated 中的可选导入标记；动作只核对固定模式、夹具身份、SHA-256
    和单组合预算，不读取夹具或运行目录。字段缺失、篡改或试图把任意路径带入任务
    事实时抛出合同错误。
    """

    if "hypothesis_import" not in value:
        return
    raw_metadata = value["hypothesis_import"]
    if not isinstance(raw_metadata, dict):
        raise ContractValidationError("TaskCreated.hypothesis_import 必须是 JSON object")
    mode = raw_metadata.get("mode")
    required_fields = {
        "mode",
        "fixture_id",
        "fixture_sha256",
        "manifest_schema",
        "expected_combination_count",
    }
    optional_fields = (
        {"structure_input"}
        if mode == "author_case3_main_hypothesis_exploration"
        else set()
    )
    metadata = require_exact_keys(
        raw_metadata,
        required_fields,
        "TaskCreated.hypothesis_import",
        optional_fields=optional_fields,
    )
    mode = metadata["mode"]
    if not isinstance(mode, str) or mode not in AUTHOR_MAIN_HYPOTHESIS_MODES:
        raise ContractValidationError("TaskCreated.hypothesis_import.mode 非法")
    if metadata["fixture_id"] != AUTHOR_MAIN_HYPOTHESIS_FIXTURE_IDS[mode]:
        raise ContractValidationError("TaskCreated.hypothesis_import.fixture_id 非法")
    if metadata["manifest_schema"] != "arche.investigation.author_hypothesis_import.v1":
        raise ContractValidationError(
            "TaskCreated.hypothesis_import.manifest_schema 非法"
        )
    if not isinstance(metadata["fixture_sha256"], str) or not _SHA256_RE.fullmatch(
        metadata["fixture_sha256"]
    ):
        raise ContractValidationError(
            "TaskCreated.hypothesis_import.fixture_sha256 必须是 64 位小写 SHA-256"
        )
    if (
        isinstance(metadata["expected_combination_count"], bool)
        or metadata["expected_combination_count"] != 1
    ):
        raise ContractValidationError(
            "作者主假设导入必须冻结 expected_combination_count=1"
        )
    if mode == "author_case3_main_hypothesis_exploration":
        try:
            # 延迟导入避免 memory.contracts 与 investigation 包初始化互相循环；
            # 校验规则的唯一实现位于 Case 3 prepared 输入专用模块。
            from chemistry_multiagent.investigation.case3_prepared_inputs import (
                validate_case3_prepared_input_metadata,
            )

            validate_case3_prepared_input_metadata(metadata.get("structure_input"))
        except (ImportError, TypeError, ValueError) as exc:
            raise ContractValidationError(
                "TaskCreated Case 3 prepared input 内容或摘要非法"
            ) from exc
    task_config = value.get("task_config")
    if not isinstance(task_config, Mapping) or task_config.get(INVESTIGATION_MODE_FIELD) != metadata["mode"]:
        raise ContractValidationError(
            "TaskCreated.hypothesis_import.mode 必须与 task_config.investigation_mode 一致"
        )
    canonical_json(dict(metadata))


def _validate_task_created_scientific_identity(value: Dict[str, Any]) -> None:
    """冻结问题、化学上下文、阶段版本和可选复现路由。"""

    require_nonempty_string(value["scientific_question"], "scientific_question")
    value["stage_architecture_version"] = validate_stage_architecture_version(
        value.get("stage_architecture_version", LEGACY_STAGE_ARCHITECTURE_VERSION)
    )
    if "case_profile" in value:
        value["case_profile"] = validate_case_profile(value["case_profile"])
    if "gaussian_route_policy" in value:
        try:
            value["gaussian_route_policy"] = validate_persisted_gaussian_route_policy(
                value["gaussian_route_policy"]
            )
        except ValueError as exc:
            raise ContractValidationError(str(exc)) from exc
    if not isinstance(value["chemistry_context"], dict):
        raise ContractValidationError("chemistry_context 必须是 JSON object")


def _validate_task_created_generation_config(value: Dict[str, Any]) -> None:
    """按照冻结的阶段版本校验 legacy 或 Investigation 任务配置。"""

    if value["stage_architecture_version"] == INVESTIGATION_STAGE_ARCHITECTURE_VERSION:
        value["task_config"] = validate_investigation_task_config(value["task_config"])
        mode = value["task_config"].get(INVESTIGATION_MODE_FIELD, "standard")
        has_import = "hypothesis_import" in value
        if mode in AUTHOR_MAIN_HYPOTHESIS_MODES and not has_import:
            raise ContractValidationError(
                "作者主假设模式必须同时保存 hypothesis_import"
            )
        if mode == "standard" and has_import:
            raise ContractValidationError(
                "standard investigation_v2 不得携带 hypothesis_import"
            )
        return
    value["task_config"] = validate_task_config(value["task_config"])


def _validate_task_created_output_identity(value: Mapping[str, Any]) -> None:
    """校验任务输出目录和工具目录哈希，拒绝机器绝对路径和非法指纹。"""

    validate_task_relative_path(
        value["task_output_relative_dir"],
        "task_output_relative_dir",
    )
    if not isinstance(value["tool_catalog_hash"], str) or not _SHA256_RE.fullmatch(
        value["tool_catalog_hash"]
    ):
        raise ContractValidationError("tool_catalog_hash 必须是 64 位小写 SHA-256")


def _validate_task_created_model_configuration(value: Dict[str, Any]) -> None:
    """校验并冻结 TaskCreated 使用的模型公开配置。"""

    model_configuration = value.get(
        "model_configuration",
        _default_model_configuration(),
    )
    value["model_configuration"] = validate_model_configuration(model_configuration)


def _validate_task_created_selection_enabled_state(
    selection: Mapping[str, Any],
    model: Mapping[str, Any],
) -> None:
    """校验专家后端选择与模型总开关是否一致。"""

    if selection["status"] == "disabled":
        if model["enabled"] is not False:
            raise ContractValidationError("disabled 专家选择只能配 enabled=false")
        return
    if model["enabled"] is not True:
        raise ContractValidationError("启用专家选择只能配 enabled=true")


def _validate_task_created_selection_provider(
    selection: Mapping[str, Any],
    model: Mapping[str, Any],
) -> None:
    """校验选择事实引用的 provider 与对应冻结角色完全一致。"""

    expected_role = {
        "arche_chem_7b": "expert",
        "cluster_expert": "cluster",
        "fallback": "fallback",
        "startup_failed": selection["failure_role"],
    }.get(selection["backend"])
    if expected_role is None:
        return
    if expected_role not in model or selection["provider"] != model[expected_role]:
        raise ContractValidationError(
            "expert_backend_selection.provider 必须等于对应候选角色配置"
        )


def _validate_task_created_selection_probe_models(
    selection: Mapping[str, Any],
    model: Mapping[str, Any],
) -> None:
    """校验成功探测返回的模型身份没有偏离 TaskCreated 候选配置。"""

    for role_name in ("expert", "cluster", "fallback", "general"):
        probe = selection["probe"].get(role_name)
        if not isinstance(probe, dict) or probe.get("reason_code") != "ok":
            continue
        if role_name not in model:
            continue
        if not model_identity_matches(
            model[role_name]["model"],
            probe.get("served_model"),
        ):
            raise ContractValidationError(
                f"expert_backend_selection.{role_name} 探测模型与候选配置不一致"
            )


def _validate_task_created_expert_backend_selection(value: Dict[str, Any]) -> None:
    """校验可选的专家后端选择及其与模型配置的绑定关系。"""

    if "expert_backend_selection" not in value:
        return
    value["expert_backend_selection"] = _validate_current_expert_runtime_selection(
        value["expert_backend_selection"]
    )
    selection = value["expert_backend_selection"]
    model = value["model_configuration"]
    _validate_task_created_selection_enabled_state(selection, model)
    _validate_task_created_selection_provider(selection, model)
    _validate_task_created_selection_probe_models(selection, model)


def validate_task_created_payload(payload: Any) -> Dict[str, Any]:
    """检查任务创建事实，并冻结所有批次共享的问题和运行配置。

    输入是 TaskCreated 的 JSON 对象；动作按问题身份、阶段配置、模型选择和输出
    引用逐段校验，并返回可持久化副本。函数不访问文件、provider 或数据库；任何
    缺字段、路径、模型身份或版本关系错误都会抛出合同错误。
    """

    value = _read_task_created_payload_mapping(payload)
    _validate_task_created_scientific_identity(value)
    _validate_task_created_generation_config(value)
    _validate_task_created_model_configuration(value)
    _validate_task_created_expert_backend_selection(value)
    _validate_task_created_hypothesis_import(value)
    _validate_task_created_output_identity(value)
    canonical_json(value)
    return dict(value)


def _validate_generation_source(value: Any, batch_index: int) -> None:
    source = require_exact_keys(
        value,
        {"kind", "previous_batch_id", "source_record_ids"},
        "generation_source",
    )
    if not isinstance(source["kind"], str) or source["kind"] not in {
        "initial",
        "working_memory_regeneration",
    }:
        raise ContractValidationError("generation_source.kind 非法")
    if not isinstance(source["source_record_ids"], list) or any(
        not isinstance(item, str) or not item for item in source["source_record_ids"]
    ):
        raise ContractValidationError("generation_source.source_record_ids 必须是字符串数组")
    if source["source_record_ids"] != sorted(set(source["source_record_ids"])):
        raise ContractValidationError("generation_source.source_record_ids 必须排序且去重")
    if batch_index == 1:
        if source != {"kind": "initial", "previous_batch_id": None, "source_record_ids": []}:
            raise ContractValidationError("初始 batch 的 generation_source 不符合固定契约")
    elif source["kind"] != "working_memory_regeneration" or not isinstance(
        source["previous_batch_id"], str
    ):
        raise ContractValidationError("后续 batch 必须直接引用上一批")


def _validate_batch_created_metadata(value: Mapping[str, Any]) -> int:
    """校验 BatchCreated 的固定字段并返回批次编号。

    输入是已确认字段集合的 BatchCreated 正文；动作检查批次、操作、因果来源和
    generation_source，不写入状态；返回正整数 batch_index，类型或来源错误时抛出
    合同错误。
    """

    require_nonempty_string(value["batch_id"], "batch_id")
    batch_index = require_json_integer(value["batch_index"], "batch_index", 1)
    require_nonempty_string(value["operation_instance_id"], "operation_instance_id")
    if value["causal_source_ref"] is not None:
        require_nonempty_string(value["causal_source_ref"], "causal_source_ref")
    _validate_generation_source(value["generation_source"], batch_index)
    return batch_index


def _validate_batch_created_question_snapshot(
    value: Mapping[str, Any],
    batch_index: int,
) -> None:
    """校验首批问题快照或后续批次的空快照。

    输入是 BatchCreated 正文及已校验的批次编号；动作核对科学问题、检索初始阶段
    和后续批次的空值约束，不修改正文；快照类型或批次位置不符时抛出合同错误。
    """

    if batch_index == 1:
        snapshot = require_exact_keys(
            value["question_snapshot"],
            {"scientific_question", "initial_query_angles", "initial_retrieval_phase"},
            "question_snapshot",
        )
        require_nonempty_string(
            snapshot["scientific_question"], "question_snapshot.scientific_question"
        )
        if not isinstance(snapshot["initial_query_angles"], list) or not isinstance(
            snapshot["initial_retrieval_phase"], dict
        ):
            raise ContractValidationError("初始 question_snapshot 类型不匹配")
    elif value["question_snapshot"] is not None:
        raise ContractValidationError("后续 batch 的 question_snapshot 必须为 null")


def _validate_batch_created_ranked_strategies(value: Mapping[str, Any]) -> list[Any]:
    """读取并校验 BatchCreated 中的 ranked_strategies 数组。

    输入是已完成字段集合检查的 BatchCreated 正文；动作只检查阶段对象、数组类型和
    非空约束，不写入状态；返回策略数组，字段缺失或为空时抛出合同错误。
    """

    phase = value["hypothesis_phase"]
    if not isinstance(phase, dict) or not isinstance(phase.get("ranked_strategies"), list):
        raise ContractValidationError("hypothesis_phase.ranked_strategies 必须是数组")
    ranked = phase["ranked_strategies"]
    if not ranked:
        raise ContractValidationError("hypothesis_phase.ranked_strategies 不能为空")
    return ranked


def _validate_batch_created_candidate_identity(
    candidate_raw: Any,
    strategy: Any,
    offset: int,
    *,
    allow_legacy_hypothesis: bool,
    hypothesis_ids: set[str],
) -> Dict[str, Any]:
    """校验一个候选的字段、策略身份和兼容 confidence。

    输入是候选、对应策略、从 1 开始的顺序编号和旧版兼容开关；动作核对候选字段、
    假设身份、顺序、排名与 confidence，并更新假设身份集合；字段漂移或重复身份
    时抛出合同错误，返回候选副本。
    """

    current_candidate_fields = {
        "hypothesis_id",
        "candidate_index",
        "rank",
        "selection_status",
        "combination_id",
        "combination_index",
    }
    candidate_fields = (
        frozenset(candidate_raw) if isinstance(candidate_raw, Mapping) else None
    )
    if candidate_fields not in {
        frozenset(current_candidate_fields),
        frozenset(current_candidate_fields | {"confidence"}),
    }:
        raise ContractValidationError(
            f"candidates[{offset - 1}] 字段不符合新协议或旧版兼容协议"
        )
    candidate = dict(candidate_raw)
    if not isinstance(strategy, dict):
        raise ContractValidationError("ranked_strategies item 必须是 object")
    validate_structured_hypothesis_claims(
        strategy,
        path=f"hypothesis_phase.ranked_strategies[{offset - 1}]",
        allow_legacy=allow_legacy_hypothesis,
    )
    hypothesis_id = require_nonempty_string(candidate["hypothesis_id"], "candidate.hypothesis_id")
    if strategy.get("hypothesis_id") != hypothesis_id:
        raise ContractValidationError("candidate.hypothesis_id 必须逐值等于 ranked_strategies 中的 ID")
    if hypothesis_id in hypothesis_ids:
        raise ContractValidationError("同一 batch 的 hypothesis_id 必须唯一")
    hypothesis_ids.add(hypothesis_id)
    if candidate["candidate_index"] != offset:
        raise ContractValidationError("candidate_index 必须从 1 连续映射 ranked_strategies")
    require_json_integer(candidate["rank"], "candidate.rank", 1)
    if "confidence" in candidate:
        confidence = candidate["confidence"]
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ContractValidationError("旧版 candidate.confidence 必须是 0 到 1 的有限数值")
    return candidate


def _validate_batch_created_candidate_selection(
    candidate: Mapping[str, Any],
    offset: int,
    *,
    selected_count: int,
    combination_ids: set[str],
    combination_indexes: set[int],
) -> None:
    """校验一个候选的 selected/not_selected 状态及组合身份。

    输入是已完成 identity 校验的候选、顺序号和选择上限；动作检查选择状态、组合
    编号和唯一性集合，不产生副作用；组合字段或状态不一致时抛出合同错误。
    """

    expected_status = "selected" if offset <= selected_count else "not_selected"
    if not isinstance(candidate["selection_status"], str):
        raise ContractValidationError("candidate.selection_status 必须是字符串枚举")
    if candidate["selection_status"] != expected_status:
        raise ContractValidationError("candidate.selection_status 与 top_n 选择不一致")
    if expected_status == "selected":
        combination_id = require_nonempty_string(candidate["combination_id"], "candidate.combination_id")
        combination_index = require_json_integer(
            candidate["combination_index"], "candidate.combination_index", 1
        )
        if combination_id in combination_ids or combination_index in combination_indexes:
            raise ContractValidationError("selected combination identity 必须唯一")
        combination_ids.add(combination_id)
        combination_indexes.add(combination_index)
    elif candidate["combination_id"] is not None or candidate["combination_index"] is not None:
        raise ContractValidationError("not_selected candidate 不能分配 combination")


def _validate_batch_created_candidate_entry(
    candidate_raw: Any,
    strategy: Any,
    offset: int,
    *,
    selected_count: int,
    allow_legacy_hypothesis: bool,
    hypothesis_ids: set[str],
    combination_ids: set[str],
    combination_indexes: set[int],
) -> None:
    """校验一个候选索引并更新本批身份集合。

    输入是候选、对应策略、顺序号和三组唯一性集合；动作串联 identity 与 selection
    校验并更新集合，不写入数据库；任一合同字段错误时抛出合同错误。
    """

    candidate = _validate_batch_created_candidate_identity(
        candidate_raw,
        strategy,
        offset,
        allow_legacy_hypothesis=allow_legacy_hypothesis,
        hypothesis_ids=hypothesis_ids,
    )
    _validate_batch_created_candidate_selection(
        candidate,
        offset,
        selected_count=selected_count,
        combination_ids=combination_ids,
        combination_indexes=combination_indexes,
    )


def _validate_batch_created_candidate_entries(
    value: Mapping[str, Any],
    ranked: list[Any],
    *,
    top_n_workflows: int,
    allow_legacy_hypothesis: bool,
    required_selected_count: int | None,
) -> None:
    """校验候选数组与策略数组逐项对应，并确认组合编号连续。

    输入是 BatchCreated 正文、已校验策略数组和任务选择上限；动作逐项调用候选校验
    并核对 v8 的五选约束，不修改持久状态；数组缺项、选择数量错误或组合编号不连
    续时抛出合同错误。
    """

    if not isinstance(value["candidates"], list) or len(value["candidates"]) != len(ranked):
        raise ContractValidationError("candidates 必须逐项映射完整 ranked_strategies")
    selected_count = min(top_n_workflows, len(ranked))
    if required_selected_count is not None:
        if required_selected_count < 1:
            raise ContractValidationError("required_selected_count 必须是正整数")
        if selected_count != required_selected_count:
            raise ContractValidationError(
                f"当前架构必须恰有 {required_selected_count} 个 selected combination"
            )
    protocol_version = value.get("investigation_protocol_version")
    if protocol_version is not None and protocol_version != INVESTIGATION_PROTOCOL_VERSION:
        raise ContractValidationError("investigation_protocol_version 未注册")
    hypothesis_ids: set[str] = set()
    combination_ids: set[str] = set()
    combination_indexes: set[int] = set()
    for offset, (candidate_raw, strategy) in enumerate(
        zip(value["candidates"], ranked), start=1
    ):
        _validate_batch_created_candidate_entry(
            candidate_raw,
            strategy,
            offset,
            selected_count=selected_count,
            allow_legacy_hypothesis=allow_legacy_hypothesis,
            hypothesis_ids=hypothesis_ids,
            combination_ids=combination_ids,
            combination_indexes=combination_indexes,
        )
    if combination_indexes != set(range(1, selected_count + 1)):
        raise ContractValidationError("combination_index 必须从 1 连续分配")


def validate_batch_created_payload(
    payload: Any,
    *,
    top_n_workflows: int,
    allow_legacy_hypothesis: bool = False,
    required_selected_count: int | None = None,
) -> Dict[str, Any]:
    """检查一批假设与候选索引是否一一对应。

    输入是 BatchCreated 正文和任务选择上限；动作校验候选、选择状态和可选的
    Investigation v8 固定五选约束；返回规范化正文，不产生副作用。字段缺失、
    选择数不符或候选正文不一致时抛出合同错误。候选只保存排名、选择状态和正式
    ID，假设正文仍以本批完整结果为准，避免同一内容在多个位置出现不一致副本。
    """

    value = require_exact_keys(
        payload,
        {
            "batch_id",
            "batch_index",
            "operation_instance_id",
            "question_snapshot",
            "generation_source",
            "hypothesis_phase",
            "candidates",
            "causal_source_ref",
        },
        "HypothesisBatchCreated.payload",
        optional_fields={"investigation_protocol_version"},
    )
    batch_index = _validate_batch_created_metadata(value)
    _validate_batch_created_question_snapshot(value, batch_index)
    ranked = _validate_batch_created_ranked_strategies(value)
    _validate_batch_created_candidate_entries(
        value,
        ranked,
        top_n_workflows=top_n_workflows,
        allow_legacy_hypothesis=allow_legacy_hypothesis,
        required_selected_count=required_selected_count,
    )
    canonical_json(value)
    return dict(value)


def validate_planning_payload(payload: Any) -> Dict[str, Any]:
    """确认入选假设已经获得身份完整、版本明确的一份工作流。"""

    value = require_exact_keys(
        payload,
        {
            "combination_id",
            "hypothesis_id",
            "workflow_id",
            "workflow_version",
            "execution_id",
            "operation_instance_id",
            "workflow",
            "causal_source_ref",
        },
        "PlanningCompleted.payload",
    )
    for field in (
        "combination_id",
        "hypothesis_id",
        "workflow_id",
        "execution_id",
        "operation_instance_id",
    ):
        require_nonempty_string(value[field], field)
    if value["workflow_version"] != 1:
        raise ContractValidationError("初始 planning 的 workflow_version 必须为 1")
    if not isinstance(value["workflow"], dict):
        raise ContractValidationError("workflow 必须是 canonical JSON object")
    if value["causal_source_ref"] is not None:
        require_nonempty_string(value["causal_source_ref"], "causal_source_ref")
    canonical_json(value)
    return dict(value)


def validate_tool_capability_gap(
    value: Any,
    *,
    path: str = "ToolCapabilityGap.payload.gap",
) -> Dict[str, Any]:
    """校验不含持久身份的能力缺口正文，供事件与结果投影共用。"""

    if not isinstance(value, dict):
        raise ContractValidationError(f"{path} 必须是 JSON object")
    legacy = "unmet_verification_requirements" in value
    expected_keys = {"result_type", "missing_capabilities", "reason"}
    if legacy:
        expected_keys.add("unmet_verification_requirements")
    gap = require_exact_keys(value, expected_keys, path)
    if gap["result_type"] != "tool_capability_gap":
        raise ContractValidationError(f"{path} 判别值不正确")
    unmet = None
    if legacy:
        unmet = validate_verification_requirements(
            gap["unmet_verification_requirements"],
            path=f"{path}.unmet_verification_requirements",
        )
    missing = gap["missing_capabilities"]
    if not isinstance(missing, list) or not 1 <= len(missing) <= 12:
        raise ContractValidationError(
            f"{path}.missing_capabilities 必须包含 1–12 条能力说明"
        )
    validated_missing: List[str] = []
    for index, item in enumerate(missing):
        if (
            not isinstance(item, str)
            or item != item.strip()
            or len(item) < 4
            or len(item) > 800
        ):
            raise ContractValidationError(
                f"{path}.missing_capabilities[{index}] 必须是规范自然语言"
            )
        validated_missing.append(item)
    if len(set(validated_missing)) != len(validated_missing):
        raise ContractValidationError(f"{path}.missing_capabilities 不得重复")
    reason = gap["reason"]
    if (
        not isinstance(reason, str)
        or reason != reason.strip()
        or len(reason) < 12
        or len(reason) > 2000
    ):
        raise ContractValidationError(f"{path}.reason 必须是现有目录无法覆盖的完整理由")
    normalized = {
        "result_type": "tool_capability_gap",
        "missing_capabilities": validated_missing,
        "reason": reason,
    }
    if legacy:
        normalized["unmet_verification_requirements"] = unmet
    canonical_json(normalized)
    return normalized


def validate_tool_capability_gap_payload(payload: Any) -> Dict[str, Any]:
    """确认 Planner 能力缺口是最小、可关联且可审计的组合级事实。"""

    value = require_exact_keys(
        payload,
        {
            "combination_id",
            "hypothesis_id",
            "operation_instance_id",
            "gap",
            "causal_source_ref",
        },
        "ToolCapabilityGap.payload",
    )
    for field in (
        "combination_id",
        "hypothesis_id",
        "operation_instance_id",
    ):
        require_nonempty_string(value[field], field)
    if value["causal_source_ref"] is not None:
        require_nonempty_string(value["causal_source_ref"], "causal_source_ref")
    normalized = dict(value)
    normalized["gap"] = validate_tool_capability_gap(value["gap"])
    canonical_json(normalized)
    return normalized


def validate_pause_payload(payload: Any) -> Dict[str, Any]:
    """确认暂停原因与范围相互匹配，并保留能够继续运行的准确位置。"""

    value = require_exact_keys(
        payload,
        {
            "combination_id",
            "operation",
            "operation_instance_id",
            "reason_code",
            "execution_id",
            "workflow_version",
            "causal_source_ref",
        },
        "ExecutionPaused.payload",
    )
    require_nonempty_string(value["operation_instance_id"], "operation_instance_id")
    scope = "task" if value["combination_id"] is None else "combination"
    if scope == "combination":
        require_nonempty_string(value["combination_id"], "combination_id")
    if (value["reason_code"], scope, value["operation"]) not in PAUSE_REASON_OPERATION_SCOPE:
        raise ContractValidationError("Pause 的 reason/scope/operation 组合不在固定矩阵中")
    if value["execution_id"] is not None:
        require_nonempty_string(value["execution_id"], "execution_id")
    if value["workflow_version"] is not None:
        require_json_integer(value["workflow_version"], "workflow_version", 1)
    if value["causal_source_ref"] is not None:
        require_nonempty_string(value["causal_source_ref"], "causal_source_ref")
    if value["operation"] in {
        "workflow_generation",
        "workflow_optimization",
        "hypothesis_generation",
    }:
        if value["execution_id"] is not None or value["workflow_version"] is not None:
            raise ContractValidationError("workflow 尚未建立的 Pause 不能伪造 execution identity")
    elif value["execution_id"] is None or value["workflow_version"] is None:
        raise ContractValidationError("workflow 建立后的 Pause 必须保留 execution identity")
    canonical_json(value)
    return dict(value)


def validate_checkpoint_payload(payload: Any) -> Dict[str, Any]:
    """核对外部作业等待记录中的身份和进度位置。

    检查点表示 Gaussian 等程序已经由监督器接管、但尚未得到最终结果。这里要求
    保存原执行、原步骤和单调递增的检查点序号，恢复时才不会误启动第二份计算。
    """

    value = require_exact_keys(
        payload,
        {
            "combination_id",
            "execution_id",
            "workflow_version",
            "step_id",
            "step_index",
            "operation_instance_id",
            "checkpoint_index",
            "causal_source_ref",
        },
        "ExecutionCheckpointed.payload",
    )
    for field in (
        "combination_id",
        "execution_id",
        "step_id",
        "operation_instance_id",
    ):
        require_nonempty_string(value[field], field)
    require_json_integer(value["workflow_version"], "workflow_version", 1)
    require_json_integer(value["step_index"], "step_index", 0)
    require_json_integer(value["checkpoint_index"], "checkpoint_index", 1)
    if value["causal_source_ref"] is not None:
        require_nonempty_string(value["causal_source_ref"], "causal_source_ref")
    canonical_json(value)
    return dict(value)


def validate_resume_payload(payload: Any) -> Dict[str, Any]:
    """确认恢复动作继续等待原来的工具调用，而不是创建新计算。

    执行身份、步骤位置、工具调用身份和来源检查点必须全部相同；任一项变化都说明
    调用方试图把另一项作业接到当前历史上，应在重新进入监督器前拒绝。
    """

    value = require_exact_keys(
        payload,
        {
            "combination_id",
            "execution_id",
            "workflow_version",
            "step_id",
            "step_index",
            "operation_instance_id",
            "checkpoint_id",
        },
        "ExecutionResumed.payload",
    )
    for field in (
        "combination_id",
        "execution_id",
        "step_id",
        "operation_instance_id",
        "checkpoint_id",
    ):
        require_nonempty_string(value[field], field)
    require_json_integer(value["workflow_version"], "workflow_version", 1)
    require_json_integer(value["step_index"], "step_index", 0)
    canonical_json(value)
    return dict(value)


def validate_attempt_payload(
    payload: Any,
    *,
    allow_legacy_router: bool = False,
) -> Dict[str, Any]:
    """验证一次完整科学尝试从执行证据到状态转换前后一致。

    Attempt 是“工作流已执行完，或在可信负向检查处结束；程序检查已完成；Router
    已作出唯一决定”的闭环记录。技术暂停不满足这些条件，因此不能借这个入口
    伪造成科学成败。
    """

    value = require_exact_keys(
        payload,
        {
            "combination_id",
            "workflow_version",
            "execution_id",
            "operation_instance_id",
            "execution_result",
            "program_validation",
            "router_decision",
            "controller_action",
            "terminal_reason",
            "causal_source_ref",
        },
        "Attempt.payload",
    )
    for field in ("combination_id", "execution_id", "operation_instance_id"):
        require_nonempty_string(value[field], field)
    require_json_integer(value["workflow_version"], "workflow_version", 1)
    for field in ("execution_result", "program_validation", "router_decision"):
        if not isinstance(value[field], dict):
            raise ContractValidationError(f"{field} 必须是 JSON object")
    router_decision = value["router_decision"]
    if not allow_legacy_router:
        require_exact_keys(
            router_decision,
            {"decision", "diagnosis"},
            "Router decision",
        )
        diagnosis = require_exact_keys(
            router_decision["diagnosis"],
            {"summary", "evidence_refs"},
            "Router diagnosis",
        )
        require_nonempty_string(diagnosis["summary"], "Router diagnosis.summary")
        evidence_refs = diagnosis["evidence_refs"]
        if (
            not isinstance(evidence_refs, list)
            or not evidence_refs
            or any(not isinstance(item, str) or not item for item in evidence_refs)
            or len(evidence_refs) != len(set(evidence_refs))
        ):
            raise ContractValidationError(
                "Router diagnosis.evidence_refs 必须是非空、互异的字符串数组"
            )
    action = value["controller_action"]
    if action not in ATTEMPT_ACTION_DECISION:
        raise ContractValidationError("controller_action 不在固定动作集合中")
    decision = router_decision.get("decision")
    if ATTEMPT_ACTION_DECISION[action] != decision:
        raise ContractValidationError("controller_action 与 Router decision 不一致")
    terminal_reason = value["terminal_reason"]
    if decision == "evidence_ready":
        if terminal_reason != "router_evidence_ready":
            raise ContractValidationError("evidence_ready 必须使用对应终态原因")
    elif decision == "revise_hypothesis":
        if terminal_reason != "router_revise_hypothesis":
            raise ContractValidationError("revise_hypothesis 必须使用对应终态原因")
    elif decision == "revise_workflow":
        if terminal_reason not in {None, "workflow_revision_limit_exhausted"}:
            raise ContractValidationError("工作流修订决定使用了不匹配的终态原因")
    if value["causal_source_ref"] is not None:
        require_nonempty_string(value["causal_source_ref"], "causal_source_ref")
    canonical_json(value)
    return dict(value)


_EXPERT_REQUEST_RECEIPT_FIELDS = {
    "schema_version",
    "status",
    "phase",
    "combination_id",
    "workflow_version",
    "execution_id",
    "operation_instance_id",
    "backend_fingerprint",
    "request_hash",
    "request_ref",
    "started_ref",
    "calls",
}


def _validate_sha256_text(value: Any, path: str) -> str:
    """校验一个不含路径和密钥的 64 位小写 SHA-256 字符串。"""

    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ContractValidationError(f"{path} 必须是 64 位小写 SHA-256")
    return value


def validate_expert_request_receipt(
    value: Any,
    *,
    path: str = "expert_request_receipt",
) -> Dict[str, Any] | None:
    """校验专家请求编译和启动收据，确保恢复能识别未知在途请求。

    收据只保存请求摘要、阶段、冻结后端指纹和每个 provider 调用的角色，不保存
    prompt 正文或认证信息。``started_ref`` 在启动收据中可以为空，在终局收据中
    必须指向任务目录内的启动文件。
    """

    if value is None:
        return None
    normalized = require_exact_keys(value, _EXPERT_REQUEST_RECEIPT_FIELDS, path)
    if normalized["schema_version"] != "expert-request-receipt.v1":
        raise ContractValidationError(f"{path}.schema_version 未注册")
    if normalized["status"] not in {"compiled", "started", "completed", "failed"}:
        raise ContractValidationError(f"{path}.status 非法")
    if normalized["phase"] not in {"gaussian_step_review", "debate"}:
        raise ContractValidationError(f"{path}.phase 非法")
    for field in ("combination_id", "operation_instance_id"):
        require_nonempty_string(normalized[field], f"{path}.{field}")
    require_json_integer(normalized["workflow_version"], f"{path}.workflow_version", 1)
    require_nonempty_string(normalized["execution_id"], f"{path}.execution_id")
    backend_fingerprint = normalized["backend_fingerprint"]
    if backend_fingerprint is not None:
        _validate_sha256_text(backend_fingerprint, f"{path}.backend_fingerprint")
    _validate_sha256_text(normalized["request_hash"], f"{path}.request_hash")
    validate_task_relative_path(normalized["request_ref"], f"{path}.request_ref")
    started_ref = normalized["started_ref"]
    if started_ref is not None:
        validate_task_relative_path(started_ref, f"{path}.started_ref")
    if normalized["status"] == "compiled" and started_ref is not None:
        raise ContractValidationError(f"{path}.compiled 不得引用 started 收据")
    if normalized["status"] == "started" and not isinstance(started_ref, str):
        raise ContractValidationError(f"{path}.started 必须引用 started 收据")
    if normalized["status"] in {"completed", "failed"} and not isinstance(started_ref, str):
        raise ContractValidationError(f"{path}.{normalized['status']} 必须引用 started 收据")
    calls = normalized["calls"]
    if not isinstance(calls, list):
        raise ContractValidationError(f"{path}.calls 必须是 array")
    if normalized["status"] == "compiled" and calls:
        raise ContractValidationError(f"{path}.compiled.calls 必须为空")
    if normalized["status"] != "compiled" and not calls:
        raise ContractValidationError(f"{path}.calls 必须是非空 array")
    allowed_calls = (
        {"gaussian_step_review"}
        if normalized["phase"] == "gaussian_step_review"
        else {
            "workflow_expert_opinion",
            "workflow_general_opinion",
            "debate_judge",
        }
    )
    call_ids: set[str] = set()
    for index, call in enumerate(calls):
        if not isinstance(call, Mapping) or set(call) != {
            "call_id",
            "call_index",
            "operation",
            "role",
            "message_hash",
            "request_hash",
            "backend_fingerprint",
        }:
            raise ContractValidationError(f"{path}.calls[{index}] 字段不完整")
        require_nonempty_string(call["call_id"], f"{path}.calls[{index}].call_id")
        if call["call_id"] in call_ids:
            raise ContractValidationError(f"{path}.calls[{index}].call_id 重复")
        call_ids.add(call["call_id"])
        if call["call_index"] != index + 1:
            raise ContractValidationError(f"{path}.calls[{index}].call_index 不连续")
        if call["operation"] not in allowed_calls:
            raise ContractValidationError(f"{path}.calls[{index}].operation 未注册")
        expected_role = (
            "expert"
            if call["operation"] in {"gaussian_step_review", "workflow_expert_opinion"}
            else "general"
        )
        if call["role"] != expected_role:
            raise ContractValidationError(f"{path}.calls[{index}].role 与 operation 不一致")
        _validate_sha256_text(call["request_hash"], f"{path}.calls[{index}].request_hash")
        _validate_sha256_text(call["message_hash"], f"{path}.calls[{index}].message_hash")
        backend_fingerprint = call["backend_fingerprint"]
        if backend_fingerprint is not None:
            _validate_sha256_text(
                backend_fingerprint,
                f"{path}.calls[{index}].backend_fingerprint",
            )
    canonical_json(dict(normalized))
    return dict(normalized)


def validate_expert_gaussian_review_payload(payload: Any) -> Dict[str, Any]:
    """校验 Planning 后专家审查及 Gaussian request 覆盖事实。"""

    required_fields = {
        "combination_id",
        "workflow_version",
        "execution_id",
        "operation_instance_id",
        "status",
        "records",
        "overrides",
        "requested_transition",
        "reason",
        "causal_source_ref",
    }
    optional_fields = {"request_receipt", "review_phase"}
    if not isinstance(payload, dict):
        raise ContractValidationError("ExpertGaussianReview.payload 必须是 JSON object")
    missing = required_fields - set(payload)
    extra = set(payload) - required_fields - optional_fields
    if missing or extra:
        raise ContractValidationError(
            "ExpertGaussianReview.payload 字段不匹配："
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    value = dict(payload)
    if "review_phase" in value:
        require_nonempty_string(value["review_phase"], "review_phase")
        if value["review_phase"] not in {"legacy", "pre_execution"}:
            raise ContractValidationError("ExpertGaussianReview.review_phase 非法")
    if "request_receipt" in value:
        value["request_receipt"] = validate_expert_request_receipt(
            value["request_receipt"],
            path="ExpertGaussianReview.request_receipt",
        )
    for field in ("combination_id", "execution_id", "operation_instance_id"):
        require_nonempty_string(value[field], field)
    require_json_integer(value["workflow_version"], "workflow_version", 1)
    if value["status"] not in {"approved", "revise", "reject", "not_configured"}:
        raise ContractValidationError("ExpertGaussianReview.status 非法")
    records = value["records"]
    if not isinstance(records, list):
        raise ContractValidationError("ExpertGaussianReview.records 必须是 array")
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ContractValidationError(f"ExpertGaussianReview.records[{index}] 必须是 object")
        if not {"step_id", "status", "reason"} <= set(record):
            raise ContractValidationError("专家审查记录缺少 step_id/status/reason")
        require_nonempty_string(record["step_id"], "ExpertGaussianReview.records.step_id")
        if record["status"] not in {"approved", "revise", "reject"}:
            raise ContractValidationError("专家审查记录 status 非法")
        require_nonempty_string(record["reason"], "ExpertGaussianReview.records.reason")
        record_base_fields = {"step_id", "status", "reason"}
        if "step_digest" in record:
            _validate_sha256_text(
                record["step_digest"],
                f"ExpertGaussianReview.records[{index}].step_digest",
            )
            record_base_fields.add("step_digest")
        if record["status"] == "revise":
            if set(record) != {*record_base_fields, "parameter_patch"}:
                raise ContractValidationError(
                    "revise 审查记录只能包含 step_id/status/reason、可选 step_digest 和 parameter_patch"
                )
            patch = record.get("parameter_patch")
            if not isinstance(patch, Mapping) or not patch:
                raise ContractValidationError(
                    "revise 审查记录必须包含非空 parameter_patch"
                )
        else:
            if set(record) != record_base_fields:
                raise ContractValidationError(
                    "approved/reject 审查记录只能包含 step_id/status/reason 和可选 step_digest"
                )
            if "replacement_step" in record:
                raise ContractValidationError(
                    "专家审查记录不得包含旧 replacement_step 字段"
                )
    overrides = value["overrides"]
    if not isinstance(overrides, Mapping) or any(
        not isinstance(key, str) or not key or not isinstance(item, Mapping)
        for key, item in overrides.items()
    ):
        raise ContractValidationError("ExpertGaussianReview.overrides 必须是 step_id 到 object 的映射")
    for step_id, patch in overrides.items():
        if "replacement_step" in patch:
            raise ContractValidationError(
                f"ExpertGaussianReview.overrides[{step_id}] 不得包含 replacement_step"
            )
        if not patch:
            raise ContractValidationError(
                f"ExpertGaussianReview.overrides[{step_id}] 必须是非空 parameter_patch"
            )
    if value.get("review_phase") == "pre_execution":
        if value["status"] == "approved" and value["requested_transition"] is not None:
            raise ContractValidationError("pre_execution approved 审查不得请求工作流修订")
        if value["status"] != "approved" and value["requested_transition"] != "revise_workflow":
            raise ContractValidationError(
                "pre_execution 非 approved 审查必须请求 revise_workflow"
            )
    elif value["status"] == "reject" and value["requested_transition"] != "revise_workflow":
        raise ContractValidationError("reject 审查必须请求 revise_workflow")
    elif value["status"] != "reject" and value["requested_transition"] is not None:
        raise ContractValidationError("非 reject 审查不得请求工作流修订")
    if value["reason"] is not None:
        require_nonempty_string(value["reason"], "ExpertGaussianReview.reason")
    if value["causal_source_ref"] is not None:
        require_nonempty_string(value["causal_source_ref"], "ExpertGaussianReview.causal_source_ref")
    canonical_json(value)
    return dict(value)


def validate_expert_debate_payload(payload: Any) -> Dict[str, Any]:
    """校验完整执行后的两份意见、五字段 Debate JSON 和最终裁判。

    新事件的 Debate 只保存科学问题、假设、两份三字段意见和程序生成的
    ``evidence_index``；旧数据库中的六字段 Gate 版本仍可读，以便迁移过程中
    不因历史事实格式而阻断恢复。新模型响应不经过旧兼容分支。
    """

    required_fields = {
        "combination_id",
        "workflow_version",
        "execution_id",
        "operation_instance_id",
        "debate",
        "judge",
        "route",
        "causal_source_ref",
    }
    optional_fields = {"request_receipt"}
    if not isinstance(payload, dict):
        raise ContractValidationError("ExpertDebate.payload 必须是 JSON object")
    missing = required_fields - set(payload)
    extra = set(payload) - required_fields - optional_fields
    if missing or extra:
        raise ContractValidationError(
            "ExpertDebate.payload 字段不匹配："
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    value = dict(payload)
    if "request_receipt" in value:
        value["request_receipt"] = validate_expert_request_receipt(
            value["request_receipt"],
            path="ExpertDebate.request_receipt",
        )
    for field in ("combination_id", "execution_id", "operation_instance_id"):
        require_nonempty_string(value[field], field)
    require_json_integer(value["workflow_version"], "workflow_version", 1)
    debate = value["debate"]
    if not isinstance(debate, Mapping):
        raise ContractValidationError("ExpertDebate.debate 必须是 object")
    debate_keys = set(debate)
    new_debate = debate_keys == {
        "scientific_question",
        "hypothesis",
        "expert_opinion",
        "general_opinion",
        "evidence_index",
    }
    legacy_debate = debate_keys == {
        "scientific_question",
        "hypothesis",
        "expert_opinion",
        "general_opinion",
        "allowed_decisions",
        "constraint_reasons",
    }
    if not (new_debate or legacy_debate):
        raise ContractValidationError(
            "ExpertDebate.debate 必须恰含五个新协议字段"
        )
    require_nonempty_string(
        debate["scientific_question"], "ExpertDebate.debate.scientific_question"
    )
    if not isinstance(debate["hypothesis"], Mapping):
        raise ContractValidationError("ExpertDebate.debate.hypothesis 必须是 object")
    allowed_decisions = None
    if legacy_debate:
        allowed_decisions = debate["allowed_decisions"]
        if (
            not isinstance(allowed_decisions, list)
            or not allowed_decisions
            or any(
                not isinstance(item, str)
                or item not in {"evidence_ready", "revise_workflow", "revise_hypothesis"}
                for item in allowed_decisions
            )
            or len(allowed_decisions) != len(set(allowed_decisions))
        ):
            raise ContractValidationError(
                "ExpertDebate.debate.allowed_decisions 不是合法的去重路由数组"
            )
        if not isinstance(debate["constraint_reasons"], list):
            raise ContractValidationError(
                "ExpertDebate.debate.constraint_reasons 必须是 array"
            )

    evidence_index = debate.get("evidence_index")
    evidence_refs: set[str] = set()
    if new_debate:
        if not isinstance(evidence_index, list) or not evidence_index:
            raise ContractValidationError("ExpertDebate.evidence_index 必须是非空 array")
        for index, item in enumerate(evidence_index, start=1):
            if not isinstance(item, Mapping) or set(item) != {
                "evidence_ref",
                "step_ref",
                "evidence_kind",
                "summary",
            }:
                raise ContractValidationError(
                    f"ExpertDebate.evidence_index[{index - 1}] 字段非法"
                )
            if item["evidence_ref"] != f"e{index}":
                raise ContractValidationError("ExpertDebate.evidence_ref 必须从 e1 连续排列")
            if item["step_ref"] is not None and (
                not isinstance(item["step_ref"], str) or not item["step_ref"]
            ):
                raise ContractValidationError("ExpertDebate.evidence_index.step_ref 非法")
            require_nonempty_string(item["evidence_kind"], "ExpertDebate.evidence_kind")
            require_nonempty_string(item["summary"], "ExpertDebate.evidence.summary")
            evidence_refs.add(item["evidence_ref"])
    for name in ("expert_opinion", "general_opinion"):
        opinion = debate[name]
        if not isinstance(opinion, Mapping) or opinion.get("opinion") not in {
            "supports_hypothesis",
            "reliable_refutation",
            "insufficient_evidence",
        }:
            raise ContractValidationError(f"ExpertDebate.debate.{name}.opinion 非法")
        if new_debate:
            if set(opinion) != {"opinion", "summary", "evidence_refs"}:
                raise ContractValidationError(
                    f"ExpertDebate.debate.{name} 必须恰含三个字段"
                )
            require_nonempty_string(opinion["summary"], f"ExpertDebate.debate.{name}.summary")
            refs = opinion["evidence_refs"]
            if (
                not isinstance(refs, list)
                or not refs
                or len(refs) != len(set(refs))
                or any(not isinstance(item, str) or item not in evidence_refs for item in refs)
            ):
                raise ContractValidationError(f"ExpertDebate.debate.{name}.evidence_refs 非法")
        else:
            if "summary" in opinion and opinion["summary"] is not None:
                require_nonempty_string(opinion["summary"], f"ExpertDebate.debate.{name}.summary")
            refs = opinion.get("evidence_refs", [])
            if not isinstance(refs, list) or any(not isinstance(item, str) or not item for item in refs):
                raise ContractValidationError(f"ExpertDebate.debate.{name}.evidence_refs 非法")
    judge = value["judge"]
    if not isinstance(judge, Mapping) or judge.get("opinion") not in {
        "supports_hypothesis",
        "reliable_refutation",
        "insufficient_evidence",
    }:
        raise ContractValidationError("ExpertDebate.judge.opinion 非法")
    if new_debate:
        if set(judge) != {"opinion", "summary", "evidence_refs"}:
            raise ContractValidationError("ExpertDebate.judge 必须恰含三个字段")
        require_nonempty_string(judge.get("summary"), "ExpertDebate.judge.summary")
        judge_refs = judge.get("evidence_refs")
        if (
            not isinstance(judge_refs, list)
            or not judge_refs
            or len(judge_refs) != len(set(judge_refs))
            or any(not isinstance(item, str) or item not in evidence_refs for item in judge_refs)
        ):
            raise ContractValidationError("ExpertDebate.judge.evidence_refs 非法")
    elif judge.get("summary") is not None:
        require_nonempty_string(judge.get("summary"), "ExpertDebate.judge.summary")
    if value["route"] not in {"evidence_ready", "revise_hypothesis", "revise_workflow"}:
        raise ContractValidationError("ExpertDebate.route 非法")
    if legacy_debate and value["route"] not in allowed_decisions:
        raise ContractValidationError(
            "ExpertDebate.route 超出 debate 中冻结的 Program Gate allowed_decisions"
        )
    if value["causal_source_ref"] is not None:
        require_nonempty_string(value["causal_source_ref"], "ExpertDebate.causal_source_ref")
    canonical_json(value)
    return dict(value)


def validate_expert_pause_payload(payload: Any) -> Dict[str, Any]:
    """校验 provider 未配置导致的专家暂停。"""

    value = require_exact_keys(
        payload,
        {
            "combination_id",
            "workflow_version",
            "execution_id",
            "operation_instance_id",
            "phase",
            "reason",
            "causal_source_ref",
        },
        "ExpertModelPaused.payload",
    )
    require_nonempty_string(value["combination_id"], "combination_id")
    require_nonempty_string(value["operation_instance_id"], "operation_instance_id")
    if value["phase"] not in {"gaussian_step_review", "debate"}:
        raise ContractValidationError("ExpertModelPaused.phase 非法")
    if value["execution_id"] is not None:
        require_nonempty_string(value["execution_id"], "execution_id")
    if value["workflow_version"] is not None:
        require_json_integer(value["workflow_version"], "workflow_version", 1)
    require_nonempty_string(value["reason"], "reason")
    if value["causal_source_ref"] is not None:
        require_nonempty_string(value["causal_source_ref"], "causal_source_ref")
    canonical_json(value)
    return dict(value)


def validate_batch_closed_payload(payload: Any) -> Dict[str, Any]:
    """确认批次结论完整覆盖每个组合，并能追溯到各自终态记录。

    一个成功组合就足以使本批成功；整批无成功结果时不得夹带结果文件。组合索引必须
    连续且唯一，避免并发结束顺序造成遗漏或同一组合被重复计入。
    """

    value = require_exact_keys(
        payload,
        {"batch_index", "combination_results", "batch_outcome"},
        "HypothesisBatchClosed.payload",
    )
    require_json_integer(value["batch_index"], "batch_index", 1)
    if value["batch_outcome"] not in {"succeeded", "failed"}:
        raise ContractValidationError("batch_outcome 必须是 succeeded 或 failed")
    results = value["combination_results"]
    if not isinstance(results, list) or not results:
        raise ContractValidationError("combination_results 必须是非空数组")
    seen_ids = set()
    seen_indexes = set()
    for offset, raw in enumerate(results):
        item = require_exact_keys(
            raw,
            {
                "combination_id",
                "combination_index",
                "terminal_state",
                "terminal_reason",
                "terminal_record_id",
                "final_result_relative_path",
            },
            f"combination_results[{offset}]",
        )
        combination_id = require_nonempty_string(
            item["combination_id"], "combination_id"
        )
        combination_index = require_json_integer(
            item["combination_index"], "combination_index", 1
        )
        if combination_id in seen_ids or combination_index in seen_indexes:
            raise ContractValidationError("批次终态组合身份或索引重复")
        seen_ids.add(combination_id)
        seen_indexes.add(combination_index)
        if item["terminal_state"] == BATCH_TECHNICAL_PAUSE_STATE:
            if (
                value["batch_outcome"] != "succeeded"
                or item["terminal_reason"] != BATCH_TECHNICAL_PAUSE_STATE
            ):
                raise ContractValidationError(
                    "technical_pause 只能随已有证据的成功批次发布"
                )
        else:
            if item["terminal_state"] not in BATCH_CLOSABLE_COMBINATION_STATES:
                raise ContractValidationError("terminal_state 不是可归约的组合终态")
            if item["terminal_reason"] not in COMBINATION_TERMINAL_REASONS:
                raise ContractValidationError("terminal_reason 不在固定集合中")
        if (
            item["terminal_state"] == "evidence_ready"
            and item["terminal_reason"] != "router_evidence_ready"
        ):
            raise ContractValidationError("evidence_ready 与 terminal_reason 不一致")
        if (
            item["terminal_state"] == "hypothesis_failed"
            and item["terminal_reason"]
            in {"router_evidence_ready", "tool_capability_gap"}
        ):
            raise ContractValidationError("hypothesis_failed 与 terminal_reason 不一致")
        if (
            item["terminal_state"] == "tool_capability_gap"
            and item["terminal_reason"] != "tool_capability_gap"
        ):
            raise ContractValidationError(
                "tool_capability_gap 与 terminal_reason 不一致"
            )
        require_nonempty_string(item["terminal_record_id"], "terminal_record_id")
        path = item["final_result_relative_path"]
        if item["terminal_state"] == "evidence_ready":
            validate_task_relative_path(path, "final_result_relative_path")
        elif path is not None:
            raise ContractValidationError("非成功组合不能发布 bounded result")
    if seen_indexes != set(range(1, len(results) + 1)):
        raise ContractValidationError("combination_results 必须按连续索引覆盖整批")
    has_evidence = any(item["terminal_state"] == "evidence_ready" for item in results)
    if (value["batch_outcome"] == "succeeded") != has_evidence:
        raise ContractValidationError("batch_outcome 与组合终态不一致")
    canonical_json(value)
    return dict(value)


def validate_revision_payload(payload: Any) -> Dict[str, Any]:
    """检查模型工作流修订记录，确保前后版本和执行身份可追溯。"""

    value = require_exact_keys(
        payload,
        {
            "combination_id",
            "operation_instance_id",
            "workflow_version_before",
            "workflow_version_after",
            "before_workflow",
            "normalized_patch",
            "after_workflow",
            "causal_source_ref",
        },
        "Revision.payload",
    )
    require_nonempty_string(value["combination_id"], "combination_id")
    require_nonempty_string(value["operation_instance_id"], "operation_instance_id")
    if value["causal_source_ref"] is not None:
        require_nonempty_string(value["causal_source_ref"], "causal_source_ref")
    before = require_json_integer(value["workflow_version_before"], "workflow_version_before", 1)
    after = require_json_integer(value["workflow_version_after"], "workflow_version_after", 2)
    if after != before + 1:
        raise ContractValidationError("workflow revision 必须恰好提高一个 version")
    for field in ("before_workflow", "normalized_patch", "after_workflow"):
        if not isinstance(value[field], dict):
            raise ContractValidationError(f"{field} 必须是 JSON object")
    canonical_json(value)
    return dict(value)


def validate_execution_step_completed_payload(payload: Any) -> Dict[str, Any]:
    """检查步骤完成事实能否独立说明它推进了哪一个执行游标。"""

    value = require_exact_keys(
        payload,
        {
            "combination_id",
            "execution_id",
            "workflow_version",
            "step_id",
            "step_index",
            "operation_instance_id",
            "causal_source_ref",
        },
        "ExecutionStepCompleted.payload",
    )
    for field in (
        "combination_id",
        "execution_id",
        "step_id",
        "operation_instance_id",
    ):
        require_nonempty_string(value[field], field)
    require_json_integer(value["workflow_version"], "workflow_version", 1)
    require_json_integer(value["step_index"], "step_index", 0)
    if value["causal_source_ref"] is not None:
        require_nonempty_string(value["causal_source_ref"], "causal_source_ref")
    canonical_json(value)
    return dict(value)


def validate_execution_request_rebound_payload(payload: Any) -> Dict[str, Any]:
    """检查 authored files 物化后把实际 request 重新绑定到当前 claim 的事实。"""

    value = require_exact_keys(
        payload,
        {
            "combination_id",
            "execution_id",
            "workflow_version",
            "step_id",
            "step_index",
            "operation_instance_id",
            "previous_request_ref",
            "request_ref",
            "causal_source_ref",
        },
        "ExecutionRequestRebound.payload",
    )
    for field in (
        "combination_id",
        "execution_id",
        "step_id",
        "operation_instance_id",
        "previous_request_ref",
        "request_ref",
    ):
        require_nonempty_string(value[field], field)
    require_json_integer(value["workflow_version"], "workflow_version", 1)
    require_json_integer(value["step_index"], "step_index", 0)
    validate_task_relative_path(value["previous_request_ref"], "previous_request_ref")
    validate_task_relative_path(value["request_ref"], "request_ref")
    if value["previous_request_ref"] == value["request_ref"]:
        raise ContractValidationError(
            "ExecutionRequestRebound 必须替换为不同的 request ref"
        )
    if value["causal_source_ref"] is not None:
        require_nonempty_string(value["causal_source_ref"], "causal_source_ref")
    canonical_json(value)
    return dict(value)


def validate_execution_step_failed_payload(payload: Any) -> Dict[str, Any]:
    """检查一次工具失败是否保存了完整工作流步骤清单。

    失败不是科学结论，也不是暂停。该事件只登记失败步骤和其后的
    ``not_executed`` 清单，供后续 Program Gate 与执行后双模型共同审查；
    清单顺序必须覆盖当前 workflow 的全部步骤。
    """

    value = require_exact_keys(
        payload,
        {
            "combination_id",
            "execution_id",
            "workflow_version",
            "step_id",
            "step_index",
            "operation_instance_id",
            "failed_step_manifest_ref",
            "step_manifest_refs",
            "causal_source_ref",
        },
        "ExecutionStepFailed.payload",
        optional_fields={"failure_diagnostic"},
    )
    for field in (
        "combination_id",
        "execution_id",
        "step_id",
        "operation_instance_id",
        "failed_step_manifest_ref",
    ):
        require_nonempty_string(value[field], field)
    require_json_integer(value["workflow_version"], "workflow_version", 1)
    require_json_integer(value["step_index"], "step_index", 0)
    refs = value["step_manifest_refs"]
    if (
        not isinstance(refs, list)
        or not refs
        or any(not isinstance(item, str) or not item for item in refs)
        or len(refs) != len(set(refs))
    ):
        raise ContractValidationError(
            "ExecutionStepFailed.step_manifest_refs 必须是非空且互异的路径数组"
        )
    for reference in refs:
        validate_task_relative_path(reference, "step_manifest_refs")
    if value["failed_step_manifest_ref"] not in refs:
        raise ContractValidationError(
            "ExecutionStepFailed.failed_step_manifest_ref 必须属于步骤清单"
        )
    if value["causal_source_ref"] is not None:
        require_nonempty_string(value["causal_source_ref"], "causal_source_ref")
    if "failure_diagnostic" in value:
        diagnostic = value["failure_diagnostic"]
        if not isinstance(diagnostic, Mapping):
            raise ContractValidationError("ExecutionStepFailed.failure_diagnostic 必须是 object")
        extra_diagnostic_fields = set(diagnostic) - {
            "code",
            "message",
            "details",
            "candidate_fixes",
        }
        if extra_diagnostic_fields:
            raise ContractValidationError(
                "failure_diagnostic 只能包含 code/message/details/candidate_fixes"
            )
        if "code" in diagnostic and diagnostic["code"] is not None:
            require_nonempty_string(diagnostic["code"], "failure_diagnostic.code")
        if "message" in diagnostic and diagnostic["message"] is not None:
            require_nonempty_string(diagnostic["message"], "failure_diagnostic.message")
        if "details" in diagnostic and not isinstance(diagnostic["details"], Mapping):
            raise ContractValidationError("failure_diagnostic.details 必须是 object")
        if "candidate_fixes" in diagnostic:
            fixes = diagnostic["candidate_fixes"]
            if not isinstance(fixes, list) or any(
                not isinstance(item, str) for item in fixes
            ):
                raise ContractValidationError(
                    "failure_diagnostic.candidate_fixes 必须是字符串数组"
                )
    canonical_json(value)
    return dict(value)


def validate_source_ref_shape(source_refs: Any) -> List[str]:
    """检查证据引用已经排序、没有重复，并且不会逃出任务目录。"""

    if not isinstance(source_refs, list) or any(
        not isinstance(item, str) or not item for item in source_refs
    ):
        raise ContractValidationError("source_refs 必须是非空字符串组成的 JSON array")
    if source_refs != sorted(set(source_refs)):
        raise ContractValidationError("source_refs 必须按 Unicode code point 排序且去重")
    for item in source_refs:
        if "/" in item:
            validate_task_relative_path(item, "source_refs artifact")
    return list(source_refs)


def validate_planning_source_refs(source_refs: Any) -> List[str]:
    """校验 Planning 的最终清单来源。

    未启用 Optimizer 的旧事件继续只引用一份清单；启用 Optimizer 时允许按
    ``generation -> optimization`` 顺序保存两份 finalized manifest，从而保留首版
    Planner 的 provenance。通用 source_refs 的排序和去重规则仍然适用；若路径带有
    两个模型操作目录，则额外核对这两个阶段的语义顺序。
    """

    refs = validate_source_ref_shape(source_refs)
    if len(refs) not in {1, 2}:
        raise ContractValidationError(
            "PlanningCompleted.source_refs 必须包含一份旧版清单或 generation、optimization 两份清单"
        )
    if len(refs) == 2:
        generation_refs = [
            item for item in refs if "/workflow_generation_inputs/" in item
        ]
        optimization_refs = [
            item for item in refs if "/workflow_optimization_inputs/" in item
        ]
        if generation_refs or optimization_refs:
            if len(generation_refs) != 1 or len(optimization_refs) != 1:
                raise ContractValidationError(
                    "PlanningCompleted 两份来源必须同时包含 generation 和 optimization manifest"
                )
            if refs.index(generation_refs[0]) > refs.index(optimization_refs[0]):
                raise ContractValidationError(
                    "PlanningCompleted.source_refs 必须按 generation、optimization 顺序排列"
                )
    return refs


def sha256_from_manifest_name(relative_path: str) -> str:
    """从最终清单文件名读取内容哈希，并确认文件名符合发布格式。"""

    name = PurePosixPath(relative_path).name
    matches = _SHA256_IN_NAME_RE.findall(name)
    if len(matches) != 1:
        raise ContractValidationError("finalized manifest 文件名必须含唯一 canonical JSON hash")
    return matches[0]


def event_key_task_created(task_id: str) -> str:
    return f"wm:task:{task_id}:created"


def event_key_batch_created(task_id: str, batch_id: str) -> str:
    return f"wm:batch:{task_id}:{batch_id}:created"


def event_key_planning(task_id: str, combination_id: str, operation_instance_id: str) -> str:
    return f"wm:planning:{task_id}:{combination_id}:{operation_instance_id}"


def event_key_expert_gaussian_review(
    task_id: str,
    combination_id: str,
    operation_instance_id: str,
) -> str:
    """唯一标识 Planning 后对同一工作流的 Gaussian 审查。"""

    return f"wm:expert-review:{task_id}:{combination_id}:{operation_instance_id}"


def event_key_expert_debate(
    task_id: str,
    combination_id: str,
    operation_instance_id: str,
) -> str:
    """唯一标识完整执行结果上的一次 debate/judge。"""

    return f"wm:expert-debate:{task_id}:{combination_id}:{operation_instance_id}"


def event_key_expert_pause(
    task_id: str,
    combination_id: str,
    operation_instance_id: str,
) -> str:
    """唯一标识专家 provider 未配置导致的结构化暂停。"""

    return f"wm:expert-pause:{task_id}:{combination_id}:{operation_instance_id}"


def event_key_retrieval_completed(task_id: str, operation_instance_id: str) -> str:
    """唯一标识一次任务级 Retrieval 成功结果。"""

    return f"wm:retrieval:{task_id}:{operation_instance_id}:completed"


def event_key_tool_capability_gap(
    task_id: str,
    combination_id: str,
    operation_instance_id: str,
) -> str:
    """让一次规划操作最多形成一条组合级工具能力缺口。"""

    return f"wm:planning-gap:{task_id}:{combination_id}:{operation_instance_id}"


def event_key_pause(task_id: str, operation_instance_id: str) -> str:
    return f"wm:pause:{task_id}:{operation_instance_id}"


def event_key_checkpoint(task_id: str, execution_id: str, checkpoint_index: int) -> str:
    """用执行身份和检查点序号标识同一外部作业的一次等待记录。"""

    return f"wm:checkpoint:{task_id}:{execution_id}:{checkpoint_index}"


def event_key_execution_step_completed(
    task_id: str,
    execution_id: str,
    operation_instance_id: str,
) -> str:
    """让同一次工具调用的成功结果只能绑定一份步骤清单。"""

    return f"wm:step:{task_id}:{execution_id}:{operation_instance_id}:completed"


def event_key_execution_step_failed(
    task_id: str,
    execution_id: str,
    operation_instance_id: str,
) -> str:
    """让同一次工具失败只能绑定一组完整失败步骤清单。"""

    return f"wm:step:{task_id}:{execution_id}:{operation_instance_id}:failed"


def event_key_execution_request_rebound(
    task_id: str,
    execution_id: str,
    operation_instance_id: str,
    request_ref: str,
) -> str:
    """用执行身份和新 request ref 标识一次物化请求 CAS 重绑定。"""

    request_digest = hashlib.sha256(request_ref.encode("utf-8")).hexdigest()
    return (
        f"wm:request-rebound:{task_id}:{execution_id}:"
        f"{operation_instance_id}:{request_digest}"
    )


def event_key_resume(task_id: str, checkpoint_id: str) -> str:
    """让同一个检查点的重复恢复请求命中同一条历史记录。"""

    return f"wm:resume:{task_id}:{checkpoint_id}"


def event_key_attempt(
    task_id: str,
    combination_id: str,
    attempt_index: int,
    workflow_version: int,
    execution_id: str,
) -> str:
    """为一次完整科学尝试生成可重复计算的稳定键。"""

    return (
        f"wm:attempt:{task_id}:{combination_id}:{attempt_index}:"
        f"{workflow_version}:{execution_id}"
    )


def event_key_batch_closed(task_id: str, batch_id: str) -> str:
    """保证每个假设批次最多发布一份关闭结论。"""

    return f"wm:batch:{task_id}:{batch_id}:closed"


def event_key_task_closed(task_id: str) -> str:
    """保证整个任务只能形成一份最终关闭事实。"""

    return f"wm:task:{task_id}:closed"


def event_key_restart(pause_id: str) -> str:
    """把一次人工恢复与它所处理的暂停记录固定绑定。"""

    return f"wm:restart:{pause_id}"


def model_revision_uuid(task_id: str, combination_id: str, operation_instance_id: str) -> str:
    """根据任务、组合和本次模型运行生成可重复计算的修订ID。"""

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"wm:model-revision:{task_id}:{combination_id}:{operation_instance_id}",
        )
    )


def event_key_investigation_session(
    task_id: str,
    batch_id: str,
    combination_id: str,
) -> str:
    """标识一个批次组合的唯一 Investigation session 身份。"""

    return f"wm:investigation-session:{task_id}:{batch_id}:{combination_id}"


def investigation_session_id(task_id: str, batch_id: str, combination_id: str) -> str:
    """返回由任务、批次和组合稳定计算的 session 身份，不依赖 SDK thread。"""

    return f"session:{task_id}:{batch_id}:{combination_id}"


def event_key_investigation_result(
    task_id: str,
    batch_id: str,
    combination_id: str,
    result_version: str,
) -> str:
    """标识一个组合的不可变 Investigation result 版本。"""

    return (
        f"wm:investigation-result:{task_id}:{batch_id}:{combination_id}:"
        f"{result_version}"
    )


def event_key_investigation_session_thread_bound(
    task_id: str,
    batch_id: str,
    combination_id: str,
) -> str:
    """标识一个 session 第一次绑定 SDK thread 的不可变事实。"""

    return f"wm:investigation-thread-bound:{task_id}:{batch_id}:{combination_id}"


def event_key_investigation_session_thread_migrated(
    task_id: str,
    batch_id: str,
    combination_id: str,
    source_codex_thread_id: str,
    target_codex_thread_id: str,
) -> str:
    """标识一次 owner/session 内的 provider thread 迁移事实。"""

    return (
        f"wm:investigation-thread-migrated:{task_id}:{batch_id}:{combination_id}:"
        f"{source_codex_thread_id}:{target_codex_thread_id}"
    )


def event_key_investigation_turn_started(
    task_id: str,
    batch_id: str,
    combination_id: str,
    turn_id: str,
) -> str:
    """标识一个 Investigation 首轮 turn 的 in-flight 事实。"""

    return f"wm:investigation-turn:{task_id}:{batch_id}:{combination_id}:{turn_id}"


def event_key_investigation_turn_waiting(
    task_id: str,
    turn_started_event_ref: str,
) -> str:
    """按被等待的 Started 事件标识一次不可变等待回执。"""

    return f"wm:investigation-turn-waiting:{task_id}:{turn_started_event_ref}"


def event_key_investigation_turn_attempt_termination(
    task_id: str,
    turn_started_event_ref: str,
) -> str:
    """按 task 和被引用 Started 事件稳定标识一次 attempt 终止事实。"""

    return f"wm:investigation-turn-attempt-terminated:{task_id}:{turn_started_event_ref}"


def event_key_investigation_legacy_waiting_resume_authorized(
    task_id: str,
    batch_id: str,
    combination_id: str,
    sidecar_record_digest: str,
) -> str:
    """按完整旧 sidecar 摘要标识一次不可变恢复授权事实。"""

    return (
        "wm:investigation-legacy-waiting-resume-authorized:"
        f"{task_id}:{batch_id}:{combination_id}:{sidecar_record_digest}"
    )


def event_key_investigation_incomplete_v2_waiting_resume_authorized(
    task_id: str,
    batch_id: str,
    combination_id: str,
    sidecar_record_digest: str,
) -> str:
    """按当前 v2 sidecar 摘要标识一次残缺 waiting 恢复授权事实。"""

    return (
        "wm:investigation-incomplete-v2-waiting-resume-authorized:"
        f"{task_id}:{batch_id}:{combination_id}:{sidecar_record_digest}"
    )


def event_key_investigation_batch_review(
    task_id: str,
    batch_id: str,
    result_version_vector: Sequence[Sequence[str]],
) -> str:
    """按完整五项结果版本向量标识一次批次 Review。"""

    vector = sorted((str(pair[0]), str(pair[1])) for pair in result_version_vector)
    digest = hashlib.sha256(canonical_json(vector).encode("utf-8")).hexdigest()
    return f"wm:investigation-review:{task_id}:{batch_id}:{digest}"


def event_key_investigation_repair_requested(
    task_id: str,
    batch_id: str,
    combination_id: str,
    repair_count: int,
) -> str:
    """标识同一组合的第 N 次持久返修请求。"""

    return f"wm:investigation-repair-requested:{task_id}:{batch_id}:{combination_id}:{repair_count}"


def event_key_investigation_repair_completed(
    task_id: str,
    request_event_ref: str,
) -> str:
    """标识一个返修请求对应的完成事实。"""

    return f"wm:investigation-repair-completed:{task_id}:{request_event_ref}"


def event_key_investigation_route_marker(
    task_id: str,
    batch_id: str,
    combination_id: str,
) -> str:
    """标识一个组合唯一的 revise_hypothesis 终局标记。"""

    return f"wm:investigation-route:{task_id}:{batch_id}:{combination_id}:revise_hypothesis"


def event_key_investigation_evidence_verification(
    task_id: str,
    batch_id: str,
    combination_id: str,
    verification_object_id: str,
) -> str:
    """标识上层权威适配器为一个组合核对的安全证据对象。"""

    return (
        f"wm:investigation-evidence:{task_id}:{batch_id}:{combination_id}:"
        f"{verification_object_id}"
    )


def event_key_task_review_session(task_id: str) -> str:
    """标识每个 task 唯一的 Review provider 会话冻结事实。"""

    return f"wm:task-review-session:{task_id}"


def event_key_task_review_turn(task_id: str, review_turn_id: str) -> str:
    """标识 Review 会话中一次 provider turn 的启动事实。"""

    return f"wm:task-review-turn:{task_id}:{review_turn_id}"


def event_key_investigation_repair_turn_started(
    task_id: str,
    request_event_ref: str,
    attempt_number: int = 1,
) -> str:
    """按 request 和 provider attempt 标识一次返修 turn 启动事实。"""

    base = f"wm:investigation-repair-turn:{task_id}:{request_event_ref}"
    return base if attempt_number == 1 else f"{base}:attempt-{attempt_number}"


def event_key_investigation_task_terminal(
    task_id: str,
    batch_id: str,
    review_event_ref: str | None = None,
) -> str:
    """标识一个 Investigation 批次产生任务终态的事实。

    首次写入保留旧版稳定键；更正后再次关闭时用新的 Review 身份生成追加键，避免
    把新的 TaskClosed 覆盖到旧事件上。输入只参与稳定字符串计算，不访问数据库。
    """

    base = f"wm:investigation-task-terminal:{task_id}:{batch_id}"
    if review_event_ref is None:
        return base
    digest = hashlib.sha256(str(review_event_ref).encode("utf-8")).hexdigest()
    return f"{base}:{digest}"


def event_key_investigation_task_correction(
    task_id: str,
    terminal_event_ref: str,
    review_event_ref: str,
    result_event_ref: str,
    combination_id: str,
) -> str:
    """按被更正的终态、Review、Result 和组合身份标识更正事实。

    修复建议不参与稳定键，因此同一身份的不同建议会触发追加器的幂等冲突，而不会
    静默产生两份互相竞争的待返修请求。函数只计算稳定键，不写入任何状态。
    """

    identity = [
        task_id,
        terminal_event_ref,
        review_event_ref,
        result_event_ref,
        combination_id,
    ]
    digest = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()
    return f"wm:investigation-task-correction:{task_id}:{digest}"


def event_key_hypothesis_selection_completed(task_id: str, batch_id: str) -> str:
    """为只完成 Top5 筛选的任务生成唯一终态事件键。"""

    return f"wm:hypothesis-selection:{task_id}:{batch_id}:completed"


def event_key_investigation_task_failure(task_id: str) -> str:
    """为一个 Investigation task 生成唯一的执行失败事件键。

    输入是已经通过任务身份校验的 task_id；动作按固定前缀拼接稳定键，不访问模型、
    文件或数据库。副作用为零，返回可用于幂等写入和重放核对的字符串；身份为空或
    不安全时由上层合同校验抛出错误，函数本身不写入任何事实。
    """

    return f"wm:investigation-task-failure:{task_id}"


# 这些包装器保持调用方可以使用“开始/记录”语义，同时明确事件 key 只有一套生成规则。
def event_key_investigation_started(task_id: str, batch_id: str, combination_id: str) -> str:
    """兼容旧调用点，返回 canonical Investigation session 事件 key。"""

    return event_key_investigation_session(task_id, batch_id, combination_id)


def event_key_batch_review(
    task_id: str,
    batch_id: str,
    result_version_vector: Sequence[Sequence[str]],
) -> str:
    """兼容旧调用点，返回 canonical 批次 Review 事件 key。"""

    return event_key_investigation_batch_review(task_id, batch_id, result_version_vector)
