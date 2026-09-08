"""把 Investigation Codex runtime 组合成 Coordinator 可注入的三个回调。

本模块只负责边界组合：冻结每个组合的结构化输入，按完整 session identity 选择
独立的 Investigation runtime，调用 prompt builder、runtime 和 result adapter，
并在结果交给 Coordinator 前执行调用方注入的事实核验器。任务级 Review 使用一份
固定 runtime/scope，只接收公开 evidence manifest 和只读 evidence tool descriptor；
本模块不会创建 MiniChem client、分配 scope 或吞掉外部异常。
"""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias

from .codex_runtime import InvestigationTurnResult
from .async_jobs import InvestigationJobIdentity, InvestigationJobState
from .contracts import require_safe_public_scientific_text
from .domain import (
    InvestigationResult,
    InvestigationSession,
    InvestigationState,
    BatchValidation,
)
from .prompts import (
    build_initial_investigation_request,
    build_investigation_job_continuation_request,
    build_investigation_job_wake_request,
    build_repair_investigation_request,
    build_task_review_request,
)
from .result_adapters import (
    adapt_investigation_turn_to_result,
    adapt_review_turn_to_batch_validation,
)


__all__ = [
    "InvestigationCombinationInput",
    "InvestigationCodexCallbackConfig",
    "InvestigationCodexCallbacks",
    "InvestigationTurnRuntimeProtocol",
    "InvestigationRuntimeProtocol",
    "InvestigationSessionIdentity",
    "TaskReviewResultBridge",
    "TaskReviewRuntimeProtocol",
]


_ENVIRONMENT_NAME = re.compile(r"[\x00-\x1f\x7f]")
_READ_ONLY_DESCRIPTOR_FORBIDDEN_TERMS = (
    "minichem",
    "submit",
    "advance",
    "execute",
    "cancel",
    "open_execution",
)


def _require_text(value: Any, label: str) -> str:
    """校验非空标识或模型名；失败时不创建可用的隐式默认值。"""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} 必须是非空字符串")
    normalized = value.strip()
    if _ENVIRONMENT_NAME.search(normalized):
        raise ValueError(f"{label} 不能包含控制字符")
    return normalized


def _freeze_json_value(value: Any, label: str) -> Any:
    """递归复制严格 JSON 值并转换为不可变的 tuple/mappingproxy。"""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} 不能包含 NaN 或无穷数字")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise TypeError(f"{label} 的 object key 必须是非空字符串")
            if key in frozen:
                raise ValueError(f"{label} 不能包含重复 object key")
            frozen[key] = _freeze_json_value(item, f"{label}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json_value(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{label} 不是严格 JSON 值")


def _freeze_json_mapping(value: Any, label: str) -> Mapping[str, Any]:
    """冻结一个结构化 JSON object；None 表示空 object。"""

    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} 必须是 JSON object")
    frozen = _freeze_json_value(value, label)
    if not isinstance(frozen, Mapping):  # pragma: no cover - 由上方校验保证
        raise TypeError(f"{label} 必须是 JSON object")
    return frozen


def _thaw_json_value(value: Any) -> Any:
    """为 prompt builder 创建独立的普通 JSON 副本，不暴露配置内部对象。"""

    if isinstance(value, Mapping):
        return {str(key): _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return copy.deepcopy(value)


@dataclass(frozen=True, slots=True)
class InvestigationSessionIdentity:
    """唯一定位一个调查组合的 task、batch 和 combination 三元组。"""

    task_id: str
    batch_id: str
    combination_id: str

    def __post_init__(self) -> None:
        """校验三个身份字段；字段不可在配置创建后修改。"""

        object.__setattr__(self, "task_id", _require_text(self.task_id, "task_id"))
        object.__setattr__(self, "batch_id", _require_text(self.batch_id, "batch_id"))
        object.__setattr__(
            self,
            "combination_id",
            _require_text(self.combination_id, "combination_id"),
        )

    @classmethod
    def from_key(cls, value: Any, label: str) -> "InvestigationSessionIdentity":
        """把配置 mapping key 解析成严格 identity；拒绝只含 combination 的别名。"""

        if isinstance(value, cls):
            return value
        if isinstance(value, tuple) and len(value) == 3:
            return cls(value[0], value[1], value[2])
        raise TypeError(
            f"{label} 必须是 InvestigationSessionIdentity 或 "
            "(task_id, batch_id, combination_id) 三元组"
        )

    def as_tuple(self) -> tuple[str, str, str]:
        """返回只读 mapping 使用的标准三元组。"""

        return self.task_id, self.batch_id, self.combination_id


_SessionIdentity: TypeAlias = tuple[str, str, str]


class InvestigationTurnRuntimeProtocol(Protocol):
    """调查 callback 所需的 runtime 最小公开接口。

    返回的 ``InvestigationTurnResult`` 可能携带真实 submit 回执；调用方必须先把它
    保存为 waiting_investigation_job，不能把空科学结果登记到证据。
    """

    def run_investigation_turn(
        self,
        request: Mapping[str, Any],
        *,
        scope_id: str,
        active_job: InvestigationJobState | Mapping[str, Any] | None = None,
        recover_existing_terminal_turn: bool = False,
    ) -> InvestigationTurnResult:
        """按指定 scope 运行首次、返修或异步作业 continuation turn。"""

class TaskReviewRuntimeProtocol(Protocol):
    """任务级 Review callback 所需的 runtime 最小公开接口。"""

    def run_task_review_turn(
        self,
        request: Mapping[str, Any],
        *,
        scope_id: str,
    ) -> InvestigationTurnResult:
        """按任务级 scope 运行只读 Review turn。"""


# 兼容已有调用方的名称；新的配置检查按两个窄协议分别执行。
InvestigationRuntimeProtocol: TypeAlias = InvestigationTurnRuntimeProtocol


class EvidenceSnapshotLeaseProtocol(Protocol):
    """只提供按已知 snapshot_id 启动并关闭 Evidence MCP 的上下文接口。"""

    def serve_evidence_snapshot(self, snapshot_id: str) -> Any:
        """启动并在退出时回收指定 snapshot 的 Evidence MCP 服务。"""


@dataclass(frozen=True, slots=True)
class InvestigationCombinationInput:
    """一个组合独占的冻结调查输入。

    结构化 hypothesis、明确条件、冻结上下文、Working Memory、calculation unit
    和工具描述都会在构造配置时深复制；后续调用方修改原始 dict/list 不会改变配置。
    Working Memory 只供 Controller 做恢复/路由和配置完整性核对，首次 Investigation
    请求不会把它投影给模型；跨批次科学经验只由 Hypothesis regeneration 使用。
    """

    hypothesis: Mapping[str, Any]
    explicit_conditions: Mapping[str, Any] = field(default_factory=dict)
    frozen_context: Mapping[str, Any] = field(default_factory=dict)
    working_memory: Mapping[str, Any] = field(default_factory=dict)
    calculation_unit: Mapping[str, Any] = field(default_factory=dict)
    initial_transition_state_structures: Mapping[str, Any] = field(default_factory=dict)
    initial_calculation_structures: Mapping[str, Any] = field(default_factory=dict)
    available_tools: Sequence[str] = ()
    tool_descriptors: Sequence[Mapping[str, Any]] = ()
    constraint_semantics: Mapping[str, Any] = field(default_factory=dict)
    workspace_input_files: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """冻结结构化 object 和工具描述，非法值在进入模型前失败。"""

        object.__setattr__(self, "hypothesis", _freeze_json_mapping(self.hypothesis, "hypothesis"))
        for field_name in (
            "explicit_conditions",
            "frozen_context",
            "working_memory",
            "calculation_unit",
            "initial_transition_state_structures",
            "initial_calculation_structures",
            "constraint_semantics",
            "workspace_input_files",
        ):
            object.__setattr__(
                self,
                field_name,
                _freeze_json_mapping(getattr(self, field_name), field_name),
            )

        raw_tools = self.available_tools
        if raw_tools is None:
            raw_tools = ()
        if isinstance(raw_tools, (str, bytes, bytearray)):
            raise TypeError("available_tools 必须是字符串数组")
        normalized_tools: list[str] = []
        for index, item in enumerate(raw_tools):
            normalized = _require_text(item, f"available_tools[{index}]")
            if normalized in normalized_tools:
                raise ValueError("available_tools 不能包含重复项")
            normalized_tools.append(normalized)
        object.__setattr__(self, "available_tools", tuple(normalized_tools))

        raw_descriptors = self.tool_descriptors
        if raw_descriptors is None:
            raw_descriptors = ()
        if isinstance(raw_descriptors, (str, bytes, bytearray)):
            raise TypeError("tool_descriptors 必须是 object 数组")
        frozen_descriptors: list[Mapping[str, Any]] = []
        for index, descriptor in enumerate(raw_descriptors):
            frozen_descriptors.append(
                _freeze_json_mapping(descriptor, f"tool_descriptors[{index}]")
            )
        object.__setattr__(self, "tool_descriptors", tuple(frozen_descriptors))


def _normalize_input_mapping(
    value: Mapping[Any, Any],
) -> dict[_SessionIdentity, InvestigationCombinationInput]:
    """冻结输入 mapping 的完整 identity 和组合输入。"""

    if not isinstance(value, Mapping) or not value:
        raise ValueError("inputs_by_session 必须是非空 mapping")
    normalized: dict[_SessionIdentity, InvestigationCombinationInput] = {}
    for index, (raw_identity, raw_input) in enumerate(value.items()):
        identity = InvestigationSessionIdentity.from_key(
            raw_identity,
            f"inputs_by_session[{index}].identity",
        ).as_tuple()
        if identity in normalized:
            raise ValueError("inputs_by_session 不能包含重复 identity")
        if isinstance(raw_input, InvestigationCombinationInput):
            combination_input = raw_input
        elif isinstance(raw_input, Mapping):
            combination_input = InvestigationCombinationInput(**dict(raw_input))
        else:
            raise TypeError(
                f"inputs_by_session[{index}] 必须是 InvestigationCombinationInput"
            )
        normalized[identity] = combination_input
    return normalized


def _normalize_runtime_mapping(
    value: Mapping[Any, Any],
) -> dict[_SessionIdentity, InvestigationRuntimeProtocol]:
    """冻结 runtime mapping 的完整 identity，拒绝隐式按组合或 scope 路由。"""

    if not isinstance(value, Mapping) or not value:
        raise ValueError("runtimes_by_session 必须是非空 mapping")
    normalized: dict[_SessionIdentity, InvestigationRuntimeProtocol] = {}
    for index, (raw_identity, runtime) in enumerate(value.items()):
        identity = InvestigationSessionIdentity.from_key(
            raw_identity,
            f"runtimes_by_session[{index}].identity",
        ).as_tuple()
        if identity in normalized:
            raise ValueError("runtimes_by_session 不能包含重复 identity")
        if not callable(getattr(runtime, "run_investigation_turn", None)):
            raise TypeError(
                f"runtimes_by_session[{index}] 缺少 run_investigation_turn"
            )
        normalized[identity] = runtime
    runtime_ids = [id(runtime) for runtime in normalized.values()]
    if len(set(runtime_ids)) != len(runtime_ids):
        raise ValueError(
            "每个 session identity 必须注入独立 Investigation runtime，不能跨 scope 复用"
        )
    return normalized


def _assert_read_only_evidence_descriptor(value: Mapping[str, Any]) -> None:
    """拒绝把 MiniChem 或可提交/推进计算的工具描述交给 Review。"""

    encoded = json.dumps(
        _thaw_json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).lower()
    if any(term in encoded for term in _READ_ONLY_DESCRIPTOR_FORBIDDEN_TERMS):
        raise ValueError("evidence_reader_tool 必须是只读 evidence 工具，不能授予 MiniChem")


def _validate_callback_dependency_contract(
    config: Any,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """校验回调依赖并冻结 Review 需要的公开 JSON 配置。

    输入是尚未冻结的 callback 配置对象；函数只核对可调用依赖、只读 evidence
    工具和可选租约，不启动 runtime、不读取外部状态。返回不可变的 evidence
    descriptor 与 review context；缺失或授予 MiniChem 写权限时抛出明确异常。
    """

    if not callable(config.evidence_snapshot_builder):
        raise TypeError("evidence_snapshot_builder 必须是可调用对象")
    if config.result_verifier is None or not callable(config.result_verifier):
        raise ValueError("必须显式注入 result_verifier，不能省略结果归属核验")
    if config.scope_execution_inventory_fence is not None and not callable(
        config.scope_execution_inventory_fence
    ):
        raise TypeError("scope_execution_inventory_fence 必须是可调用对象")
    if config.repair_result_reference_provider is not None and not callable(
        config.repair_result_reference_provider
    ):
        raise TypeError("repair_result_reference_provider 必须是可调用对象")
    if config.review_thread_id_provider is not None and not callable(
        config.review_thread_id_provider
    ):
        raise TypeError("review_thread_id_provider 必须是可调用对象")
    if config.scope_execution_inventory_fence is not None and (
        config.repair_result_reference_provider is None
    ):
        raise ValueError(
            "启用 scope_execution_inventory_fence 时必须显式提供"
            " repair_result_reference_provider"
        )
    if config.evidence_lease is not None and not callable(
        getattr(config.evidence_lease, "serve_evidence_snapshot", None)
    ):
        raise TypeError("evidence_lease 必须提供 serve_evidence_snapshot")
    if not isinstance(config.evidence_reader_tool, Mapping):
        raise TypeError("evidence_reader_tool 必须是 JSON object")

    frozen_descriptor = _freeze_json_mapping(
        config.evidence_reader_tool,
        "evidence_reader_tool",
    )
    if not frozen_descriptor:
        raise ValueError("evidence_reader_tool 不能为空")
    _assert_read_only_evidence_descriptor(frozen_descriptor)
    return frozen_descriptor, _freeze_json_mapping(
        config.review_context,
        "review_context",
    )


def _normalize_callback_identity_mappings(
    inputs_by_session: Mapping[Any, InvestigationCombinationInput],
    runtimes_by_session: Mapping[Any, InvestigationRuntimeProtocol],
    task_id: str,
) -> tuple[
    dict[_SessionIdentity, InvestigationCombinationInput],
    dict[_SessionIdentity, InvestigationRuntimeProtocol],
]:
    """规范化完整 session identity mapping 并核对任务归属。

    输入是 callback 配置中的组合输入、独立 runtime 和已规范化 task_id；函数不
    调用 runtime 或修改外部 mapping，只返回可冻结的普通字典。key 集合不一致或
    任一 identity 属于其它任务时抛出 ValueError。
    """

    normalized_inputs = _normalize_input_mapping(inputs_by_session)
    normalized_runtimes = _normalize_runtime_mapping(runtimes_by_session)
    if set(normalized_inputs) != set(normalized_runtimes):
        missing_inputs = sorted(set(normalized_runtimes) - set(normalized_inputs))
        missing_runtimes = sorted(set(normalized_inputs) - set(normalized_runtimes))
        raise ValueError(
            "inputs_by_session 与 runtimes_by_session identity mapping 不一致: "
            f"missing_inputs={missing_inputs}, missing_runtimes={missing_runtimes}"
        )
    if any(identity[0] != task_id for identity in normalized_inputs):
        raise ValueError("所有组合 identity 必须属于配置 task_id")
    return normalized_inputs, normalized_runtimes


def _validate_task_review_runtime_contract(
    review_runtime: TaskReviewRuntimeProtocol | None,
    runtimes_by_session: Mapping[_SessionIdentity, InvestigationRuntimeProtocol],
) -> None:
    """核对 task Review runtime 具备 Review 入口且不复用组合 runtime。

    输入是任务级 Review runtime 和已规范化的组合 runtime mapping；函数不执行
    任何模型调用。缺少 Review 方法或复用任一 Investigation runtime 时抛出 TypeError
    或 ValueError。
    """

    if review_runtime is None:
        # Investigation-only Harness 只需要首次/返修回调；task-level Review 在
        # 结果形成后由独立 Review adapter 单独装配。保留 None 是为了避免在
        # Investigation seam 中创建第二套 Review runtime。
        return
    if not callable(getattr(review_runtime, "run_task_review_turn", None)):
        raise TypeError("review_runtime 缺少 run_task_review_turn")
    if id(review_runtime) in {id(runtime) for runtime in runtimes_by_session.values()}:
        raise ValueError("Review runtime 必须与组合 Investigation runtime 分离")


@dataclass(frozen=True, slots=True)
class InvestigationCodexCallbackConfig:
    """三个 Coordinator 回调共用的显式冻结配置。

    ``inputs_by_session`` 与 ``runtimes_by_session`` 必须使用完整的
    ``(task_id, batch_id, combination_id)`` identity 且 key 集合完全相同；不存在
    按 combination、scope 或当前状态猜测的回退。模型名、Review scope、evidence
    snapshot builder、只读 evidence descriptor 和结果核验器全部必须显式注入；完整
    Review coordinator 必须提供 Review runtime，Investigation-only seam 可明确传入
    ``None``，但调用 Review 方法时会失败。
    """

    task_id: str
    scientific_question: str
    inputs_by_session: Mapping[Any, InvestigationCombinationInput]
    runtimes_by_session: Mapping[Any, InvestigationRuntimeProtocol]
    # Investigation-only Harness 不装配 Review runtime；完整 coordinator 才提供
    # task-level Review。None 不是隐式降级，调用 Review 方法时会明确失败。
    review_runtime: TaskReviewRuntimeProtocol | None
    review_scope_id: str
    evidence_snapshot_builder: Callable[
        [InvestigationState, tuple[InvestigationResult, ...], tuple[tuple[str, str], ...]],
        Mapping[str, Any] | Any,
    ]
    evidence_reader_tool: Mapping[str, Any]
    result_verifier: Callable[
        [InvestigationSession, InvestigationResult], InvestigationResult | None
    ] | None
    investigation_model: str
    review_model: str
    review_context: Mapping[str, Any] = field(default_factory=dict)
    scope_execution_inventory_fence: Callable[
        [InvestigationSession, Sequence[str], Sequence[str]], Mapping[str, Any]
    ] | None = None
    repair_result_reference_provider: Callable[
        [InvestigationSession], tuple[Sequence[str], Sequence[str]]
    ] | None = None
    review_thread_id_provider: Callable[[], str | None] | None = None
    evidence_lease: EvidenceSnapshotLeaseProtocol | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """冻结文本、依赖和 identity mapping，构成三个回调共用配置。"""

        task_id = _require_text(self.task_id, "task_id")
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(
            self,
            "scientific_question",
            require_safe_public_scientific_text(
                self.scientific_question,
                label="scientific_question",
            ),
        )
        object.__setattr__(
            self,
            "investigation_model",
            _require_text(self.investigation_model, "investigation_model"),
        )
        object.__setattr__(
            self,
            "review_model",
            _require_text(self.review_model, "review_model"),
        )
        object.__setattr__(
            self,
            "review_scope_id",
            _require_text(self.review_scope_id, "review_scope_id"),
        )
        frozen_descriptor, frozen_context = _validate_callback_dependency_contract(self)
        normalized_inputs, normalized_runtimes = _normalize_callback_identity_mappings(
            self.inputs_by_session,
            self.runtimes_by_session,
            task_id,
        )
        _validate_task_review_runtime_contract(
            self.review_runtime,
            normalized_runtimes,
        )
        object.__setattr__(self, "evidence_reader_tool", frozen_descriptor)
        object.__setattr__(self, "review_context", frozen_context)
        object.__setattr__(self, "inputs_by_session", MappingProxyType(normalized_inputs))
        object.__setattr__(self, "runtimes_by_session", MappingProxyType(normalized_runtimes))


class InvestigationCodexCallbacks:
    """可直接注入 ``run_investigation_and_experiment_batch`` 的回调集合。

    首次调查固定产生 ``v1``；返修只恢复原组合 thread，并用当前结果版本与原始
    advice 构造请求。下一版本依据当前 Result 的 ``vN`` 推导为 ``vN+1``，同时
    兼容投影在 ``RepairTurnStarted`` 窗口暂存的启动次数。Review 以当前状态的
    完整五项结果和版本向量生成公开证据快照，再沿任务级固定 thread/scope 调用
    runtime。任何 identity、version、vector、snapshot、runtime 或 verifier 异常
    都直接向上抛出，调用方可把三个公开属性传给 Coordinator。
    """

    def __init__(self, config: InvestigationCodexCallbackConfig) -> None:
        """保存已冻结配置并执行一次 runtime/依赖合同检查。"""

        if not isinstance(config, InvestigationCodexCallbackConfig):
            raise TypeError("config 必须是 InvestigationCodexCallbackConfig")
        self.config = config

    def _resolve_session(
        self,
        session: InvestigationSession,
    ) -> tuple[InvestigationCombinationInput, InvestigationRuntimeProtocol]:
        """按完整 session identity 查找输入和 runtime，拒绝任何降级匹配。"""

        if not isinstance(session, InvestigationSession):
            raise TypeError("session 必须是 InvestigationSession")
        if session.task_id != self.config.task_id:
            raise ValueError("session identity 与 callback 配置不匹配")
        identity = (session.task_id, session.batch_id, session.combination_id)
        inputs = self.config.inputs_by_session.get(identity)
        runtime = self.config.runtimes_by_session.get(identity)
        if inputs is None or runtime is None:
            raise ValueError(
                "session identity 没有精确匹配的组合输入和 Investigation runtime"
            )
        return inputs, runtime

    @staticmethod
    def _verify_result(
        session: InvestigationSession,
        result: InvestigationResult,
        verifier: Callable[
            [InvestigationSession, InvestigationResult], InvestigationResult | None
        ],
    ) -> InvestigationResult:
        """执行调用方结果归属核验，并拒绝核验器改变绑定身份或科学事实。"""

        # 不捕获 verifier 异常：未终态、跨 scope 或证据归属错误必须原样到达上层。
        verified = verifier(session, result)
        if verified is None:
            return result
        if not isinstance(verified, InvestigationResult):
            raise TypeError("result_verifier 必须返回 InvestigationResult 或 None")
        if verified != result:
            raise ValueError("result_verifier 不能改变未经核验的 InvestigationResult")
        if (
            verified.task_id != session.task_id
            or verified.batch_id != session.batch_id
            or verified.combination_id != session.combination_id
            or verified.scope_id != session.scope_id
        ):
            raise ValueError("result_verifier 返回的结果身份与 session 不一致")
        return verified

    def _build_initial_investigation_request(
        self,
        session: InvestigationSession,
        inputs: InvestigationCombinationInput,
    ) -> Mapping[str, Any]:
        """按冻结组合输入构造原始 initial request，供正常和 recovery-only 共用。

        输入是已通过完整 identity 校验的 session 与组合输入；动作只复制结构化科学
        字段并生成 request envelope，不启动 runtime、Controller 或 MiniChem。返回值
        可交给同一组合的 runtime；字段缺失或模型上下文非法时由 prompt builder 失败。
        """

        return build_initial_investigation_request(
            session.task_id,
            session.batch_id,
            session.combination_id,
            self.config.scientific_question,
            _thaw_json_value(inputs.hypothesis),
            explicit_conditions=_thaw_json_value(inputs.explicit_conditions),
            frozen_context=_thaw_json_value(inputs.frozen_context),
            calculation_unit=_thaw_json_value(inputs.calculation_unit),
            initial_transition_state_structures=(
                _thaw_json_value(inputs.initial_transition_state_structures)
                if inputs.initial_transition_state_structures
                else None
            ),
            initial_calculation_structures=(
                _thaw_json_value(inputs.initial_calculation_structures)
                if inputs.initial_calculation_structures
                else None
            ),
            constraint_semantics=(
                _thaw_json_value(inputs.constraint_semantics)
                if inputs.constraint_semantics
                else None
            ),
            workspace_input_files=(
                _thaw_json_value(inputs.workspace_input_files)
                if inputs.workspace_input_files
                else None
            ),
            available_tools=tuple(inputs.available_tools),
            tool_descriptors=tuple(
                _thaw_json_value(descriptor)
                for descriptor in inputs.tool_descriptors
            ),
            codex_thread_id=session.codex_thread_id,
            model=self.config.investigation_model,
        )

    def _build_repair_investigation_request(
        self,
        session: InvestigationSession,
        inputs: InvestigationCombinationInput,
        repair_advice: str,
    ) -> Mapping[str, Any]:
        """按当前返修状态重建原始 DeepSeek request，供正常和恢复共用。

        输入是已经通过完整 identity 校验的 ``repair_pending`` session、冻结组合
        输入和 Working Memory 当前 advice；动作只重建 envelope，不启动 runtime、
        provider 或 MiniChem。返回 request 的结果版本、返修编号和 thread 必须与
        当前 session 一致，字段非法时直接失败而不猜测旧请求。
        """

        if session.phase != "repair_pending":
            raise ValueError("DeepSeek 返修只接受 repair_pending session")
        if session.codex_thread_id is None or session.result_version is None:
            raise ValueError("返修 session 必须绑定当前 result_version 和 Codex thread")
        version_match = re.fullmatch(r"v([1-9][0-9]*)", session.result_version)
        if version_match is None:
            raise ValueError("返修 session 的当前 result_version 格式非法")
        current_version_number = int(version_match.group(1))
        if current_version_number > 3:
            raise ValueError("返修 session 的当前 result_version 已超过返修上限")
        if session.repair_count not in {
            current_version_number - 1,
            current_version_number,
        }:
            raise ValueError("返修 session 的 repair_count 与当前 result_version 不一致")
        return build_repair_investigation_request(
            session.task_id,
            session.batch_id,
            session.combination_id,
            session.codex_thread_id,
            session.result_version,
            repair_advice,
            repair_count=current_version_number,
            constraint_semantics=(
                _thaw_json_value(inputs.constraint_semantics)
                if inputs.constraint_semantics
                else None
            ),
            model=self.config.investigation_model,
        )

    def _build_investigation_job_continuation_request(
        self,
        session: InvestigationSession,
        inputs: InvestigationCombinationInput,
        waiting: InvestigationJobState,
    ) -> Mapping[str, Any]:
        """用等待 sidecar 的完整身份重建唯一 continuation request。

        输入是当前 session、冻结组合输入和已核对的 waiting sidecar；动作只调用
        与正常 continuation 相同的 prompt builder，不访问 Controller、MiniChem 或
        provider。返回的 metadata.request_kind 和 prompt 内容必须与正常交接路径
        完全一致，供 recovery-only 严格复用磁盘 request-intent digest。
        """

        return build_investigation_job_continuation_request(
            session.task_id,
            session.batch_id,
            session.combination_id,
            waiting.codex_thread_id,
            job_id=waiting.identity.job_id,
            operation_id=waiting.identity.operation_id,
            submission_key=waiting.identity.submission_key,
            action_id=waiting.identity.action_id,
            backend_id=waiting.identity.backend_id,
            job_family=waiting.job_family,
            job_type=waiting.identity.job_type,
            runtime=waiting.identity.runtime,
            software_id=waiting.identity.software_id,
            operation_fingerprint=waiting.identity.operation_fingerprint,
            scope_id=waiting.scope_id,
            scientific_question=self.config.scientific_question,
            hypothesis=_thaw_json_value(inputs.hypothesis),
            explicit_conditions=_thaw_json_value(inputs.explicit_conditions),
            frozen_context=_thaw_json_value(inputs.frozen_context),
            calculation_unit=_thaw_json_value(inputs.calculation_unit),
            initial_transition_state_structures=(
                _thaw_json_value(inputs.initial_transition_state_structures)
                if inputs.initial_transition_state_structures
                else None
            ),
            initial_calculation_structures=(
                _thaw_json_value(inputs.initial_calculation_structures)
                if inputs.initial_calculation_structures
                else None
            ),
            constraint_semantics=(
                _thaw_json_value(inputs.constraint_semantics)
                if inputs.constraint_semantics
                else None
            ),
            workspace_input_files=(
                _thaw_json_value(inputs.workspace_input_files)
                if inputs.workspace_input_files
                else None
            ),
            available_tools=self._build_wake_available_tools(inputs, waiting),
            tool_descriptors=tuple(
                _thaw_json_value(descriptor)
                for descriptor in inputs.tool_descriptors
            ),
            model=self.config.investigation_model,
        )

    def _build_recovery_investigation_request(
        self,
        session: InvestigationSession,
        inputs: InvestigationCombinationInput,
        active_job: InvestigationJobState | Mapping[str, Any],
        repair_advice: str | None,
        request_kind: str,
    ) -> Mapping[str, Any]:
        """按 lifecycle 判定的 request kind 重建 repair 或 continuation 请求。

        recovery-only 的 request digest 必须对应原始 intent。因而后续 repair attempt
        不能沿用 deepseek_repair prompt；它要复用正常 continuation 的 builder。该
        helper 只处理 request 构造和身份核对，不执行任何外部调用。
        """

        if session.phase != "repair_pending":
            raise ValueError("repair recovery 只接受 repair_pending session")
        if request_kind not in {
            "deepseek_repair",
            "investigation_job_continuation",
        }:
            raise ValueError("repair recovery request_kind 不受支持")
        if request_kind == "deepseek_repair":
            if not isinstance(repair_advice, str) or not repair_advice.strip():
                raise ValueError("repair recovery 缺少当前 repair_advice")
            return self._build_repair_investigation_request(
                session,
                inputs,
                repair_advice,
            )
        try:
            waiting = (
                active_job
                if isinstance(active_job, InvestigationJobState)
                else InvestigationJobState.from_mapping(active_job)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("continuation recovery waiting identity 非法") from exc
        if waiting.status != "waiting_investigation_job":
            raise ValueError("continuation recovery 只接受 waiting_investigation_job")
        if (
            waiting.task_id != session.task_id
            or waiting.batch_id != session.batch_id
            or waiting.combination_id != session.combination_id
            or waiting.scope_id != session.scope_id
            or session.codex_thread_id != waiting.codex_thread_id
        ):
            raise ValueError("continuation recovery waiting identity 与 session 不一致")
        return self._build_investigation_job_continuation_request(
            session,
            inputs,
            waiting,
        )

    def run_initial_investigation_turn(
        self,
        session: InvestigationSession,
        active_job: InvestigationJobState | Mapping[str, Any] | None = None,
        recover_existing_terminal_turn: bool = False,
    ) -> InvestigationTurnResult:
        """构造并运行首次调查 turn；真实 submit 回执由 runtime 附在返回值上。"""

        inputs, runtime = self._resolve_session(session)
        if session.phase != "created":
            raise ValueError("首次 Investigation 只接受 created session")
        request = self._build_initial_investigation_request(session, inputs)
        if self.config.scope_execution_inventory_fence is not None and active_job is None:
            # created session 的 fence 必须明确要求空 job/operation；调用方不能
            # 借输入 mapping 或模型正文隐式推导一个已有副作用清单。
            self.config.scope_execution_inventory_fence(session, (), ())
        if active_job is None and not recover_existing_terminal_turn:
            return runtime.run_investigation_turn(request, scope_id=session.scope_id)
        runtime_arguments: dict[str, Any] = {
            "scope_id": session.scope_id,
            "active_job": active_job,
        }
        if recover_existing_terminal_turn:
            runtime_arguments["recover_existing_terminal_turn"] = True
        return runtime.run_investigation_turn(request, **runtime_arguments)

    def has_recoverable_interrupted_investigation_turn(
        self,
        session: InvestigationSession,
        active_job: InvestigationJobState | Mapping[str, Any],
        *,
        repair_advice: str | None = None,
        request_kind: str | None = None,
    ) -> bool:
        """只读检查旧 initial 或 repair request-intent 是否可在原 thread 上收束。

        输入是 session、同组合 waiting sidecar 和（repair_pending 时）当前 WM advice；
        动作构造对应原始 request，再调用 runtime 的本地 recovery 检查。不会调用模型、
        Controller 或 MiniChem；缺少 runtime 窄接口时返回 False，身份损坏由 runtime
        抛出。
        """

        inputs, runtime = self._resolve_session(session)
        if session.phase == "created":
            if repair_advice is not None or request_kind is not None:
                raise ValueError("created session 不能带 repair recovery 参数")
            request = self._build_initial_investigation_request(session, inputs)
        elif session.phase == "repair_pending":
            normalized_request_kind = (
                "deepseek_repair" if request_kind is None else request_kind
            )
            request = self._build_recovery_investigation_request(
                session,
                inputs,
                active_job,
                repair_advice,
                normalized_request_kind,
            )
        else:
            return False
        checker = getattr(
            runtime,
            "has_recoverable_interrupted_investigation_turn",
            None,
        )
        if not callable(checker):
            return False
        return bool(
            checker(
                request,
                scope_id=session.scope_id,
                active_job=active_job,
            )
        )

    def recover_interrupted_investigation_turn(
        self,
        session: InvestigationSession,
        active_job: InvestigationJobState | Mapping[str, Any],
        *,
        repair_advice: str | None = None,
        request_kind: str | None = None,
    ) -> None:
        """收束同一 session 的旧 interrupted initial 或 repair turn，不启动新 turn。

        输入是 session、经核对的 waiting sidecar 和（repair_pending 时）当前 WM advice；
        动作复用同一组合 runtime 的 recovery-only 入口，先由 adapter journal/history
        核对旧 turn，再把终态生命周期写入 Working Memory。该阶段不提交、轮询、收集、
        取消或清理作业。
        """

        inputs, runtime = self._resolve_session(session)
        if session.phase == "created":
            if repair_advice is not None or request_kind is not None:
                raise ValueError("created session 不能带 repair recovery 参数")
            request = self._build_initial_investigation_request(session, inputs)
        elif session.phase == "repair_pending":
            normalized_request_kind = (
                "deepseek_repair" if request_kind is None else request_kind
            )
            request = self._build_recovery_investigation_request(
                session,
                inputs,
                active_job,
                repair_advice,
                normalized_request_kind,
            )
        else:
            raise ValueError("recovery-only 只接受 created 或 repair_pending session")
        recover = getattr(runtime, "recover_interrupted_investigation_turn", None)
        if not callable(recover):
            raise ValueError("Investigation runtime 缺少 recovery-only 接口")
        recover(
            request,
            scope_id=session.scope_id,
            active_job=active_job,
        )

    def run_initial_investigation(self, session: InvestigationSession) -> InvestigationResult:
        """构造并运行一个组合的首次调查，然后返回已核验的 v1 结果。"""

        turn_result = self.run_initial_investigation_turn(session)
        result = adapt_investigation_turn_to_result(session, "v1", turn_result)
        return self._verify_result(session, result, self.config.result_verifier)

    def run_investigation_job_wake(
        self,
        session: InvestigationSession,
        waiting: InvestigationJobState,
        terminal_status: str,
        terminal_message: str | None = None,
        wake_request: Mapping[str, Any] | None = None,
    ) -> InvestigationTurnResult:
        """在原 Codex thread 上发送 Controller 终态 wake turn。

        输入是当前 session、已持久化的 ``waiting_investigation_job`` 和 Controller
        确认的最小终态；首次唤醒构造 wake prompt，恢复唤醒则复用已经持久化的完整
        request envelope。动作只调用原组合 runtime，不查询或收集 MiniChem payload。
        返回最终科学 turn 或下一次真实 submit 回执；终态后由调用方再用
        ``adapt_investigation_turn_to_result`` 登记科学结果；身份漂移、非等待状态或
        换 thread 会 fail-closed。
        """

        if not isinstance(waiting, InvestigationJobState):
            raise TypeError("waiting 必须是 InvestigationJobState")
        inputs, runtime = self._resolve_session(session)
        if waiting.status != "waiting_investigation_job":
            raise ValueError("wake 只接受 waiting_investigation_job")
        if (
            waiting.task_id != session.task_id
            or waiting.batch_id != session.batch_id
            or waiting.combination_id != session.combination_id
        ):
            raise ValueError("waiting job identity 与 session 不一致")
        if session.codex_thread_id != waiting.codex_thread_id:
            allows_migration = getattr(runtime, "allows_provider_thread_migration", None)
            if not callable(allows_migration) or not bool(allows_migration()):
                raise ValueError("wake 必须复用原 Codex thread，除非显式启用 provider migration")
        if wake_request is None:
            request = build_investigation_job_wake_request(
                session.task_id,
                session.batch_id,
                session.combination_id,
                waiting.codex_thread_id,
                job_id=waiting.identity.job_id,
                operation_id=waiting.identity.operation_id,
                submission_key=waiting.identity.submission_key,
                action_id=waiting.identity.action_id,
                backend_id=waiting.identity.backend_id,
                terminal_status=terminal_status,
                job_family=waiting.job_family,
                job_type=waiting.identity.job_type,
                runtime=waiting.identity.runtime,
                software_id=waiting.identity.software_id,
                operation_fingerprint=waiting.identity.operation_fingerprint,
                scope_id=waiting.scope_id,
                terminal_message=terminal_message,
                scientific_question=self.config.scientific_question,
                hypothesis=_thaw_json_value(inputs.hypothesis),
                explicit_conditions=_thaw_json_value(inputs.explicit_conditions),
                frozen_context=_thaw_json_value(inputs.frozen_context),
                calculation_unit=_thaw_json_value(inputs.calculation_unit),
                initial_transition_state_structures=(
                    _thaw_json_value(inputs.initial_transition_state_structures)
                    if inputs.initial_transition_state_structures
                    else None
                ),
                initial_calculation_structures=(
                    _thaw_json_value(inputs.initial_calculation_structures)
                    if inputs.initial_calculation_structures
                    else None
                ),
                constraint_semantics=(
                    _thaw_json_value(inputs.constraint_semantics)
                    if inputs.constraint_semantics
                    else None
                ),
                workspace_input_files=(
                    _thaw_json_value(inputs.workspace_input_files)
                    if inputs.workspace_input_files
                    else None
                ),
                available_tools=self._build_wake_available_tools(inputs, waiting),
                tool_descriptors=tuple(
                    _thaw_json_value(descriptor)
                    for descriptor in inputs.tool_descriptors
                ),
                model=self.config.investigation_model,
            )
        else:
            request = self._validate_persisted_wake_request(
                wake_request,
                session=session,
                waiting=waiting,
                terminal_status=terminal_status,
                constraint_semantics=inputs.constraint_semantics,
                workspace_input_files=inputs.workspace_input_files,
            )
            if not self._wake_request_has_scientific_context(
                request,
                inputs.initial_transition_state_structures,
                inputs.initial_calculation_structures,
                inputs.constraint_semantics,
                inputs.calculation_unit,
                inputs.workspace_input_files,
            ):
                request = build_investigation_job_wake_request(
                    session.task_id,
                    session.batch_id,
                    session.combination_id,
                    session.codex_thread_id or waiting.codex_thread_id,
                    job_id=waiting.identity.job_id,
                    operation_id=waiting.identity.operation_id,
                    submission_key=waiting.identity.submission_key,
                    action_id=waiting.identity.action_id,
                    backend_id=waiting.identity.backend_id,
                    terminal_status=terminal_status,
                    job_family=waiting.job_family,
                    job_type=waiting.identity.job_type,
                    runtime=waiting.identity.runtime,
                    software_id=waiting.identity.software_id,
                    operation_fingerprint=waiting.identity.operation_fingerprint,
                    scope_id=waiting.scope_id,
                    scientific_question=self.config.scientific_question,
                    hypothesis=_thaw_json_value(inputs.hypothesis),
                    explicit_conditions=_thaw_json_value(inputs.explicit_conditions),
                    frozen_context=_thaw_json_value(inputs.frozen_context),
                    calculation_unit=_thaw_json_value(inputs.calculation_unit),
                    initial_transition_state_structures=(
                        _thaw_json_value(inputs.initial_transition_state_structures)
                        if inputs.initial_transition_state_structures
                        else None
                    ),
                    initial_calculation_structures=(
                        _thaw_json_value(inputs.initial_calculation_structures)
                        if inputs.initial_calculation_structures
                        else None
                    ),
                    constraint_semantics=(
                        _thaw_json_value(inputs.constraint_semantics)
                        if inputs.constraint_semantics
                        else None
                    ),
                    workspace_input_files=(
                        _thaw_json_value(inputs.workspace_input_files)
                        if inputs.workspace_input_files
                        else None
                    ),
                    available_tools=self._build_wake_available_tools(inputs, waiting),
                    tool_descriptors=tuple(
                        _thaw_json_value(descriptor)
                        for descriptor in inputs.tool_descriptors
                    ),
                    model=self.config.investigation_model,
                )
        # 把等待状态原样交给 runtime；provider migration 需要据此核对旧 thread
        # 和唯一 job identity，并只允许新 thread 查询/收集该作业。
        return runtime.run_investigation_turn(
            request,
            scope_id=session.scope_id,
            active_job=waiting,
        )

    def run_investigation_job_continuation(
        self,
        session: InvestigationSession,
        waiting: InvestigationJobState,
    ) -> InvestigationTurnResult:
        """在任意作业阶段继续同一 Codex thread，让模型自行查询和收集。

        输入是当前 Investigation session 和持久 waiting sidecar；动作只构造一个
        continuation request 并调用原组合 runtime，不调用 Controller status/arm，也不
        提交作业。模型可以先列出当前 scope 的全部作业，再决定快速查询 Python 或约每
        按提交后约 1、5、10、20、30 分钟查询、之后每约 30 分钟持续查询 Gaussian；runtime 通过 active_job 身份禁止换作业和重复提交。
        """

        if not isinstance(waiting, InvestigationJobState):
            raise TypeError("waiting 必须是 InvestigationJobState")
        inputs, runtime = self._resolve_session(session)
        if waiting.status != "waiting_investigation_job":
            raise ValueError("continuation 只接受 waiting_investigation_job")
        if (
            waiting.task_id != session.task_id
            or waiting.batch_id != session.batch_id
            or waiting.combination_id != session.combination_id
            or waiting.scope_id != session.scope_id
        ):
            raise ValueError("waiting job identity 与 session 不一致")
        if session.codex_thread_id != waiting.codex_thread_id:
            allows_migration = getattr(runtime, "allows_provider_thread_migration", None)
            if not callable(allows_migration) or not bool(allows_migration()):
                raise ValueError(
                    "continuation 必须复用原 Codex thread，除非显式启用 provider migration"
                )
        request = self._build_investigation_job_continuation_request(
            session,
            inputs,
            waiting,
        )
        return runtime.run_investigation_turn(
            request,
            scope_id=session.scope_id,
            active_job=waiting,
        )

    @staticmethod
    def _wake_request_has_scientific_context(
        request: Mapping[str, Any],
        initial_transition_state_structures: Mapping[str, Any] | None = None,
        initial_calculation_structures: Mapping[str, Any] | None = None,
        constraint_semantics: Mapping[str, Any] | None = None,
        calculation_unit: Mapping[str, Any] | None = None,
        workspace_input_files: Mapping[str, Any] | None = None,
    ) -> bool:
        """检查 wake user JSON 是否包含清空历史所需的科学上下文。"""

        messages = request.get("messages")
        if not isinstance(messages, Sequence) or len(messages) < 2:
            return False
        content = messages[-1].get("content") if isinstance(messages[-1], Mapping) else None
        if not isinstance(content, str):
            return False
        try:
            payload = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            return False
        required = {
            "scientific_question",
            "explicit_conditions",
            "hypothesis",
            "frozen_context",
        }
        if calculation_unit is None or calculation_unit:
            required.add("calculation_unit")
        if initial_transition_state_structures:
            required.add("initial_transition_state_structures")
        if initial_calculation_structures:
            required.add("initial_calculation_structures")
        if constraint_semantics:
            required.add("constraint_semantics")
        if workspace_input_files:
            required.add("workspace_input_files")
        return isinstance(payload, Mapping) and required.issubset(payload)

    @staticmethod
    def _build_wake_available_tools(
        inputs: InvestigationCombinationInput,
        waiting: InvestigationJobState,
    ) -> tuple[str, ...]:
        """合并冻结工具与等待作业的查询/收集工具，并按首次出现顺序去重。"""

        return tuple(
            dict.fromkeys(
                (
                    *inputs.available_tools,
                    waiting.submit_tool,
                    waiting.status_tool,
                    waiting.collect_tool,
                    "list_execution_jobs",
                )
            )
        )

    @staticmethod
    def _validate_persisted_wake_request(
        wake_request: Mapping[str, Any],
        *,
        session: InvestigationSession,
        waiting: InvestigationJobState,
        terminal_status: str,
        constraint_semantics: Mapping[str, Any] | None = None,
        workspace_input_files: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """核对并复制持久 wake envelope，防止恢复时改换 job、thread 或终态。"""

        request = _freeze_json_mapping(wake_request, "wake_request")
        thawed = _thaw_json_value(request)
        if not isinstance(thawed, dict):  # pragma: no cover - 冻结函数已保证
            raise ValueError("wake_request 必须是 JSON object")
        if thawed.get("thread_id") != waiting.codex_thread_id:
            raise ValueError("持久 wake request 的 thread 与等待状态不一致")
        metadata = thawed.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("持久 wake request 缺少 metadata")
        expected_metadata = {
            "request_kind": "investigation_job_wake",
            "research_task_id": session.task_id,
            "investigation_context_id": session.batch_id,
            "hypothesis_execution_input_record_id": session.combination_id,
            "codex_thread_id": waiting.codex_thread_id,
            "job_id": waiting.identity.job_id,
            "operation_id": waiting.identity.operation_id,
            "submission_key": waiting.identity.submission_key,
            "terminal_status": terminal_status,
            "job_family": waiting.job_family,
        }
        if waiting.identity.action_id is not None:
            expected_metadata["action_id"] = waiting.identity.action_id
        if waiting.identity.backend_id is not None:
            expected_metadata["backend_id"] = waiting.identity.backend_id
        for field_name in (
            "job_type",
            "runtime",
            "software_id",
            "operation_fingerprint",
        ):
            expected_value = getattr(waiting.identity, field_name)
            if expected_value is not None:
                expected_metadata[field_name] = expected_value
        for field_name, expected_value in expected_metadata.items():
            if metadata.get(field_name) != expected_value:
                raise ValueError(
                    f"持久 wake request 的 {field_name} 与等待状态不一致"
                )
        identity_values = {
            "job_id": metadata.get("job_id"),
            "operation_id": metadata.get("operation_id"),
            "submission_key": metadata.get("submission_key"),
            "job_family": metadata.get("job_family"),
        }
        for field_name in (
            "action_id",
            "backend_id",
            "job_type",
            "runtime",
            "software_id",
            "operation_fingerprint",
        ):
            if field_name in metadata:
                identity_values[field_name] = metadata[field_name]
        try:
            persisted_identity = InvestigationJobIdentity.from_mapping(
                identity_values,
                label="持久 wake request identity",
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("持久 wake request 的作业身份非法") from exc
        if persisted_identity.as_mapping() != waiting.identity.as_mapping():
            raise ValueError("持久 wake request 的完整作业身份与等待状态不一致")
        messages = thawed.get("messages")
        payload = (
            messages[-1].get("content")
            if isinstance(messages, list)
            and messages
            and isinstance(messages[-1], Mapping)
            else None
        )
        if constraint_semantics:
            try:
                decoded_payload = json.loads(payload)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError("持久 wake request 的科学上下文不是 JSON") from exc
            if not isinstance(decoded_payload, Mapping) or decoded_payload.get(
                "constraint_semantics"
            ) != _thaw_json_value(constraint_semantics):
                raise ValueError("持久 wake request 的 constraint_semantics 与任务不一致")
        if workspace_input_files:
            try:
                decoded_payload = json.loads(payload)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError("持久 wake request 的科学上下文不是 JSON") from exc
            if (
                isinstance(decoded_payload, Mapping)
                and "workspace_input_files" in decoded_payload
                and decoded_payload["workspace_input_files"]
                != _thaw_json_value(workspace_input_files)
            ):
                raise ValueError("持久 wake request 的 workspace 输入路径与任务不一致")
        return thawed

    def run_deepseek_repair_turn(
        self,
        session: InvestigationSession,
        repair_advice: str,
    ) -> InvestigationTurnResult:
        """恢复组合原 thread 运行 DeepSeek 返修 turn；submit 回执由 runtime 捕获。

        ``repair_count`` 是已经登记完成的返修结果数。恢复时可能已经写入
        ``RepairTurnStarted``，但对应 Result 尚未完成；此时旧投影曾短暂把启动次数
        写进该字段。当前 Result 版本和真实 turn 身份才是下一版编号的依据，允许这
        两种投影在同一真实 turn 上恢复，禁止把仍为 ``v1`` 的结果误认成 ``v2``。
        """

        inputs, runtime = self._resolve_session(session)
        request = self._build_repair_investigation_request(
            session,
            inputs,
            repair_advice,
        )
        if self.config.scope_execution_inventory_fence is not None:
            provider = self.config.repair_result_reference_provider
            if provider is None:  # pragma: no cover - 构造器已 fail-closed
                raise ValueError("repair fence 缺少当前结果引用 provider")
            references = provider(session)
            if (
                not isinstance(references, Sequence)
                or isinstance(references, (str, bytes, bytearray))
                or len(references) != 2
            ):
                raise ValueError("repair_result_reference_provider 必须返回二元引用序列")
            expected_jobs, expected_operations = references
            if any(
                isinstance(values, (str, bytes, bytearray))
                or not isinstance(values, Sequence)
                for values in (expected_jobs, expected_operations)
            ):
                raise ValueError("repair fence 引用必须是 job/operation 序列")
            self.config.scope_execution_inventory_fence(
                session,
                tuple(expected_jobs),
                tuple(expected_operations),
            )
        return runtime.run_investigation_turn(request, scope_id=session.scope_id)

    def run_deepseek_repair(
        self,
        session: InvestigationSession,
        repair_advice: str,
    ) -> InvestigationResult:
        """恢复组合原 thread 进行 DeepSeek 返修，并返回确定版本的已核验结果。"""

        turn_result = self.run_deepseek_repair_turn(session, repair_advice)
        current_version = session.result_version
        if current_version is None:
            raise ValueError("返修 session 缺少当前 result_version")
        version_match = re.fullmatch(r"v([1-9][0-9]*)", current_version)
        if version_match is None:
            raise ValueError("返修 session 的当前 result_version 格式非法")
        next_version = f"v{int(version_match.group(1)) + 1}"
        result = adapt_investigation_turn_to_result(session, next_version, turn_result)
        return self._verify_result(session, result, self.config.result_verifier)

    @staticmethod
    def _normalize_review_inputs(
        state: InvestigationState,
        current_results: Sequence[InvestigationResult],
        result_version_vector: Sequence[tuple[str, str]],
    ) -> tuple[tuple[InvestigationResult, ...], tuple[tuple[str, str], ...]]:
        """严格核对五项当前结果与版本向量，并按 state session 顺序重建。"""

        if not isinstance(state, InvestigationState):
            raise TypeError("state 必须是 InvestigationState")
        InvestigationCodexCallbacks._validate_review_collection_sizes(
            current_results,
            result_version_vector,
        )
        normalized_vector = InvestigationCodexCallbacks._normalize_review_vector(
            result_version_vector
        )
        expected_vector = InvestigationCodexCallbacks._expected_review_vector(state)
        if dict(normalized_vector) != dict(expected_vector):
            raise ValueError("result_version_vector 与当前 state 版本漂移")
        by_combination = InvestigationCodexCallbacks._index_review_results(
            state,
            current_results,
        )
        ordered_results = InvestigationCodexCallbacks._order_review_results(
            state,
            by_combination,
        )
        return ordered_results, expected_vector

    @staticmethod
    def _project_schema_v10_review_results(
        results: Sequence[InvestigationResult],
    ) -> tuple[dict[str, Any], ...]:
        """把 schema-v10 Result 显式投影成当前 Review 请求字段。"""

        return tuple(
            {
                "research_task_id": result.task_id,
                "investigation_context_id": result.batch_id,
                "hypothesis_execution_input_record_id": result.combination_id,
                "result_version": result.result_version,
                "self_judgment": result.self_judgment,
                "narrative": result.narrative,
                "job_refs": list(result.job_refs),
                "operation_refs": list(result.operation_refs),
                "artifact_refs": list(result.artifact_refs),
                "observations": list(result.observations),
                "limitations": list(result.limitations),
            }
            for result in results
        )

    @staticmethod
    def _project_schema_v10_evidence_snapshot(
        raw_snapshot: Mapping[str, Any],
    ) -> dict[str, Any]:
        """把 schema-v10 Evidence manifest 显式投影成当前 Review 字段。"""

        snapshot = _thaw_json_value(
            _freeze_json_mapping(raw_snapshot, "evidence_snapshot")
        )
        if not isinstance(snapshot, dict):  # pragma: no cover - 冻结函数已保证
            raise TypeError("evidence_snapshot 必须是 JSON object")
        canonical_identity = {
            "research_task_id",
            "investigation_context_id",
        }
        legacy_identity = {"task_id", "batch_id"}
        if canonical_identity.issubset(snapshot):
            if legacy_identity.intersection(snapshot):
                raise ValueError("evidence_snapshot 不能混用新旧身份字段")
        elif legacy_identity.issubset(snapshot):
            snapshot["research_task_id"] = snapshot.pop("task_id")
            snapshot["investigation_context_id"] = snapshot.pop("batch_id")
        else:
            raise ValueError("evidence_snapshot 缺少任务或 Context 身份")
        raw_pairs = snapshot.get("result_pairs")
        if isinstance(raw_pairs, (str, bytes, bytearray)) or not isinstance(
            raw_pairs,
            Sequence,
        ):
            raise TypeError("evidence_snapshot.result_pairs 必须是数组")
        pairs: list[dict[str, Any]] = []
        for index, raw_pair in enumerate(raw_pairs):
            if not isinstance(raw_pair, Mapping):
                raise TypeError(
                    f"evidence_snapshot.result_pairs[{index}] 必须是 object"
                )
            pair = dict(raw_pair)
            if "hypothesis_execution_input_record_id" in pair:
                if "combination_id" in pair:
                    raise ValueError("evidence result pair 不能混用新旧身份字段")
            elif "combination_id" in pair:
                pair["hypothesis_execution_input_record_id"] = pair.pop(
                    "combination_id"
                )
            else:
                raise ValueError("evidence result pair 缺少执行输入身份")
            pairs.append(pair)
        snapshot["result_pairs"] = pairs
        return snapshot

    @staticmethod
    def _validate_review_collection_sizes(
        current_results: Sequence[InvestigationResult],
        result_version_vector: Sequence[tuple[str, str]],
    ) -> None:
        """校验 Review 输入只能是标准五项或作者实验单项序列。"""

        if isinstance(current_results, (str, bytes, bytearray)) or not isinstance(
            current_results, Sequence
        ):
            raise TypeError("current_results 必须是 InvestigationResult 数组")
        if len(current_results) not in {1, 5}:
            raise ValueError("current_results 必须包含一项或五项")
        if isinstance(result_version_vector, (str, bytes, bytearray)) or not isinstance(
            result_version_vector, Sequence
        ):
            raise TypeError("result_version_vector 必须是一项或五项二元组数组")
        if len(result_version_vector) not in {1, 5}:
            raise ValueError("result_version_vector 必须包含一项或五项")

    @staticmethod
    def _normalize_review_vector(
        result_version_vector: Sequence[tuple[str, str]],
    ) -> tuple[tuple[str, str], ...]:
        """规范化 Review 版本向量并拒绝重复组合。"""

        normalized: list[tuple[str, str]] = []
        for index, pair in enumerate(result_version_vector):
            if isinstance(pair, (str, bytes, bytearray)) or not isinstance(pair, Sequence):
                raise TypeError(f"result_version_vector[{index}] 必须是二元组")
            if len(pair) != 2:
                raise ValueError(f"result_version_vector[{index}] 必须是二元组")
            normalized.append(
                (
                    _require_text(pair[0], f"result_version_vector[{index}].combination_id"),
                    _require_text(pair[1], f"result_version_vector[{index}].result_version"),
                )
            )
        if len({item[0] for item in normalized}) != len(normalized):
            raise ValueError("result_version_vector 的组合 identity 不能重复")
        return tuple(normalized)

    @staticmethod
    def _expected_review_vector(
        state: InvestigationState,
    ) -> tuple[tuple[str, str], ...]:
        """读取当前 session 的完整版本向量。"""

        expected = tuple(
            (session.combination_id, session.result_version or "")
            for session in state.sessions
        )
        if any(not version for _, version in expected):
            raise ValueError("当前 session 都必须绑定 result_version")
        return expected

    @staticmethod
    def _index_review_results(
        state: InvestigationState,
        current_results: Sequence[InvestigationResult],
    ) -> dict[str, InvestigationResult]:
        """按组合 identity 索引 Review 输入结果。"""

        by_combination: dict[str, InvestigationResult] = {}
        for index, result in enumerate(current_results):
            if not isinstance(result, InvestigationResult):
                raise TypeError(f"current_results[{index}] 必须是 InvestigationResult")
            if result.task_id != state.task_id or result.batch_id != state.batch_id:
                raise ValueError("current_results 与 state task/batch identity 漂移")
            if result.combination_id in by_combination:
                raise ValueError("current_results 不能重复 combination identity")
            by_combination[result.combination_id] = result
        return by_combination

    @staticmethod
    def _order_review_results(
        state: InvestigationState,
        by_combination: Mapping[str, InvestigationResult],
    ) -> tuple[InvestigationResult, ...]:
        """按 state session 顺序核对结果版本、scope、thread 和已保存事实。"""

        ordered_results: list[InvestigationResult] = []
        for session in state.sessions:
            result = by_combination.get(session.combination_id)
            if result is None:
                raise ValueError("current_results 未完整覆盖 state session identity")
            if result.result_version != session.result_version:
                raise ValueError("current_results 与 state result_version 漂移")
            saved_result = next(
                (
                    item
                    for item in state.results
                    if item.combination_id == result.combination_id
                    and item.result_version == result.result_version
                ),
                None,
            )
            if saved_result is None or saved_result != result:
                raise ValueError("current_results 不是 state 中已保存的当前结果")
            if result.scope_id != session.scope_id:
                raise ValueError("current_results 与 session scope 漂移")
            if (
                session.codex_thread_id is None
                or result.codex_thread_id != session.codex_thread_id
            ):
                raise ValueError("current_results 与 session Codex thread 漂移")
            ordered_results.append(result)
        if set(by_combination) != {session.combination_id for session in state.sessions}:
            raise ValueError("current_results 含有 state 之外的 combination identity")
        return tuple(ordered_results)

    def _existing_review_thread_id(self, state: InvestigationState) -> str | None:
        """读取当前和历史验证记录中的唯一 Review thread，拒绝线程分叉。"""

        thread_ids = [validation.review_thread_id for validation in state.validations]
        for history in state.history:
            thread_ids.extend(
                validation.review_thread_id for validation in history.validations
            )
        if not thread_ids:
            return None
        if len(set(thread_ids)) != 1:
            raise ValueError("已有 Review thread identity 漂移")
        state_thread_id = thread_ids[0]
        if self.config.review_thread_id_provider is None:
            return state_thread_id
        provided_thread_id = self.config.review_thread_id_provider()
        if provided_thread_id is not None and provided_thread_id != state_thread_id:
            raise ValueError("Working Memory Review thread 与历史 Review thread 漂移")
        return state_thread_id

    def _read_review_thread_id(self, state: InvestigationState) -> str | None:
        """从 Working Memory 或当前领域快照恢复 task 级 Review thread。"""

        state_thread_id = self._existing_review_thread_id(state)
        if state_thread_id is not None:
            return state_thread_id
        if self.config.review_thread_id_provider is None:
            return None
        provided_thread_id = self.config.review_thread_id_provider()
        if provided_thread_id is not None and not isinstance(provided_thread_id, str):
            raise TypeError("review_thread_id_provider 必须返回字符串或 None")
        return provided_thread_id

    def run_task_batch_review(
        self,
        state: InvestigationState,
        current_results: Sequence[InvestigationResult],
        result_version_vector: Sequence[tuple[str, str]],
    ) -> BatchValidation:
        """创建公开五项 evidence 快照，运行任务级 Review 并适配批次验证。"""

        if not isinstance(state, InvestigationState):
            raise TypeError("state 必须是 InvestigationState")
        if state.task_id != self.config.task_id:
            raise ValueError("Review state task identity 与 callback 配置不匹配")
        if self.config.review_runtime is None:
            raise RuntimeError(
                "Investigation-only callbacks 不提供 task-level Review runtime"
            )
        ordered_results, ordered_vector = self._normalize_review_inputs(
            state,
            current_results,
            result_version_vector,
        )
        raw_snapshot = self.config.evidence_snapshot_builder(
            state,
            ordered_results,
            ordered_vector,
        )
        if hasattr(raw_snapshot, "to_public_manifest") and callable(
            raw_snapshot.to_public_manifest
        ):
            raw_snapshot = raw_snapshot.to_public_manifest()
        if not isinstance(raw_snapshot, Mapping):
            raise TypeError("evidence_snapshot_builder 必须返回严格 public manifest")
        snapshot = self._project_schema_v10_evidence_snapshot(raw_snapshot)
        public_results = self._project_schema_v10_review_results(ordered_results)
        review_thread_id = self._read_review_thread_id(state)
        request = build_task_review_request(
            research_task_id=state.task_id,
            investigation_context_id=state.batch_id,
            scientific_question=self.config.scientific_question,
            current_results=public_results,
            evidence_snapshot=snapshot,
            evidence_reader_tool=_thaw_json_value(self.config.evidence_reader_tool),
            review_context=_thaw_json_value(self.config.review_context),
            model=self.config.review_model,
            review_thread_id=review_thread_id,
        )
        snapshot_id = snapshot.get("snapshot_id")
        if not isinstance(snapshot_id, str) or not snapshot_id.strip():
            raise ValueError("evidence snapshot 必须显式包含 snapshot_id")
        lease_context = (
            self.config.evidence_lease.serve_evidence_snapshot(snapshot_id)
            if self.config.evidence_lease is not None
            else nullcontext()
        )
        with lease_context:
            turn_result = self.config.review_runtime.run_task_review_turn(
                request,
                scope_id=self.config.review_scope_id,
            )
        return adapt_review_turn_to_batch_validation(
            state,
            ordered_vector,
            turn_result,
        )

    @property
    def initial_investigator(self) -> Callable[[InvestigationSession], InvestigationResult]:
        """返回可直接注入 Coordinator 的首次调查器。"""

        return self.run_initial_investigation

    @property
    def repair_investigator(
        self,
    ) -> Callable[[InvestigationSession, str], InvestigationResult]:
        """返回可直接注入 Coordinator 的返修调查器。"""

        return self.run_deepseek_repair

    @property
    def batch_reviewer(
        self,
    ) -> Callable[
        [InvestigationState, tuple[InvestigationResult, ...], tuple[tuple[str, str], ...]],
        BatchValidation,
    ]:
        """返回可直接注入 Coordinator 的批次 Review 器。"""

        return self.run_task_batch_review


class TaskReviewResultBridge:
    """只装配 task-level Review 的公开 Result bridge。

    该 bridge 不要求 Investigation runtime/input mapping，因此可以在 Harness
    已经交付 V2 Result 后单独复用原 Review thread 和 Evidence lease。它复用同一
    evidence snapshot、request 和 turn-result 适配合同，但不会创建 Investigation
    session，也不会解释 thread/turn/job 等 Harness 私有状态。
    """

    def __init__(
        self,
        *,
        task_id: str,
        scientific_question: str,
        review_runtime: TaskReviewRuntimeProtocol,
        review_scope_id: str,
        evidence_snapshot_builder: Callable[..., Mapping[str, Any]],
        evidence_reader_tool: Mapping[str, Any],
        review_model: str,
        review_context: Mapping[str, Any] | None = None,
        review_thread_id_provider: Callable[[], str | None] | None = None,
        evidence_lease: EvidenceSnapshotLeaseProtocol | None = None,
    ) -> None:
        """保存独立 Review 依赖并校验最小公开接口。"""

        self.task_id = _require_text(task_id, "task_id")
        self.scientific_question = require_safe_public_scientific_text(
            scientific_question,
            label="scientific_question",
        )
        if not callable(getattr(review_runtime, "run_task_review_turn", None)):
            raise TypeError("review_runtime 必须提供 run_task_review_turn")
        if not callable(evidence_snapshot_builder):
            raise TypeError("evidence_snapshot_builder 必须可调用")
        self.review_runtime = review_runtime
        self.review_scope_id = _require_text(review_scope_id, "review_scope_id")
        self.evidence_snapshot_builder = evidence_snapshot_builder
        self.evidence_reader_tool = copy.deepcopy(dict(evidence_reader_tool))
        self.review_model = _require_text(review_model, "review_model")
        self.review_context = copy.deepcopy(dict(review_context or {}))
        self.review_thread_id_provider = review_thread_id_provider
        self.evidence_lease = evidence_lease

    @staticmethod
    def _read_review_thread_id(
        state: InvestigationState,
        provider: Callable[[], str | None] | None,
    ) -> str | None:
        """从已有验证事实和只读 provider 恢复唯一 Review thread。"""

        thread_ids = [validation.review_thread_id for validation in state.validations]
        for history in state.history:
            thread_ids.extend(
                validation.review_thread_id for validation in history.validations
            )
        if thread_ids:
            if len(set(thread_ids)) != 1:
                raise ValueError("已有 Review thread identity 漂移")
            thread_id = thread_ids[0]
            provided = provider() if provider is not None else None
            if provided is not None and provided != thread_id:
                raise ValueError("Working Memory Review thread 与历史 Review thread 漂移")
            return thread_id
        if provider is None:
            return None
        provided = provider()
        if provided is not None and not isinstance(provided, str):
            raise TypeError("review_thread_id_provider 必须返回字符串或 None")
        return provided

    def run_task_batch_review(
        self,
        state: InvestigationState,
        current_results: Sequence[InvestigationResult],
        result_version_vector: Sequence[tuple[str, str]],
    ) -> BatchValidation:
        """用既有 task Review runtime 审查当前 V2 Result 对应的公开快照。"""

        if not isinstance(state, InvestigationState) or state.task_id != self.task_id:
            raise ValueError("Review state task identity 与 bridge 配置不匹配")
        ordered_results, ordered_vector = InvestigationCodexCallbacks._normalize_review_inputs(
            state,
            current_results,
            result_version_vector,
        )
        raw_snapshot = self.evidence_snapshot_builder(
            state,
            ordered_results,
            ordered_vector,
        )
        if hasattr(raw_snapshot, "to_public_manifest") and callable(
            raw_snapshot.to_public_manifest
        ):
            raw_snapshot = raw_snapshot.to_public_manifest()
        if not isinstance(raw_snapshot, Mapping):
            raise TypeError("evidence_snapshot_builder 必须返回严格 public manifest")
        snapshot = InvestigationCodexCallbacks._project_schema_v10_evidence_snapshot(
            raw_snapshot
        )
        public_results = InvestigationCodexCallbacks._project_schema_v10_review_results(
            ordered_results
        )
        request = build_task_review_request(
            research_task_id=state.task_id,
            investigation_context_id=state.batch_id,
            scientific_question=self.scientific_question,
            current_results=public_results,
            evidence_snapshot=snapshot,
            evidence_reader_tool=_thaw_json_value(self.evidence_reader_tool),
            review_context=_thaw_json_value(self.review_context),
            model=self.review_model,
            review_thread_id=self._read_review_thread_id(
                state,
                self.review_thread_id_provider,
            ),
        )
        snapshot_id = snapshot.get("snapshot_id")
        if not isinstance(snapshot_id, str) or not snapshot_id.strip():
            raise ValueError("evidence snapshot 必须显式包含 snapshot_id")
        lease_context = (
            self.evidence_lease.serve_evidence_snapshot(snapshot_id)
            if self.evidence_lease is not None
            else nullcontext()
        )
        with lease_context:
            turn_result = self.review_runtime.run_task_review_turn(
                request,
                scope_id=self.review_scope_id,
            )
        return adapt_review_turn_to_batch_validation(
            state,
            ordered_vector,
            turn_result,
        )
