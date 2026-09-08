#!/usr/bin/env python3
"""
Chemistry Multi-Agent System Controller - 计算化学多智能体系统控制器

使用冻结的 MiniChem Action/Backend 目录，按固定顺序编排一次 Investigation and Experiment 任务：
1. Retrieval：检索文献并整理科学上下文。
2. Hypothesis：生成、比较并筛选科学假设。
3. Hypothesis Confirmation：研究员确认需要执行的 1–5 条假设，并可提供最终修改和材料。
4. Investigation and Experiment：每个已确认假设进入独立 harness，使用 MiniChem 自主设计并执行验证实验。
5. Task Review：DeepSeek 按完整结果版本向量审查证据，要求 repair、revise_hypothesis 或终局回答。
6. Publish：发布成功结果、批次结论或结构化暂停说明。

Memory On 使用 Working Memory 保存可恢复事实；Memory Off 的 v2 入口当前明确拒绝启动，
不会静默回到旧 Planner/Executor。上述阶段顺序和 MiniChem 工具入口保持一致。
"""

from __future__ import annotations

import os
import json
import logging
import argparse
import copy
from contextlib import ExitStack
import re
import stat
import uuid
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Dict, List, Any, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

from chemistry_multiagent.controllers.investigation_v2_production import (
    ResearchTaskProductionAssembly,
    ResearchTaskProductionError,
    V2_TASK_CONTRACT_VERSION,
)
from chemistry_multiagent.agents.retrieval_agent import (
    RetrievalAgent,
    normalize_paper_search_config,
)
from chemistry_multiagent.agents.hypothesis_agent import (
    HypothesisAgent,
    RankedHypothesisGenerator,
    project_scientific_hypothesis_identity,
)
from chemistry_multiagent.agents.expert_model_agent import (
    ExpertModelAgent,
    ExpertModelConfig,
    ExpertProvider,
)
from chemistry_multiagent.memory import ResearchTaskWorkingMemory
from chemistry_multiagent.memory.contracts import (
    SUPPORTED_CASE_PROFILES,
    ContractValidationError,
    validate_investigation_task_config,
    validate_task_config,
    validate_case_profile,
    validate_model_configuration,
    validate_expert_backend_selection,
)
from chemistry_multiagent.investigation.author_hypothesis_import import (
    CASE1_AUTHOR_MAIN_HYPOTHESIS_MODE,
    CASE3_AUTHOR_MAIN_HYPOTHESIS_MODE,
    CASE3_AUTHOR_MAIN_HYPOTHESIS_ADVISORY_MODE,
    CASE3_AUTHOR_MAIN_HYPOTHESIS_EXPLORATION_MODE,
    load_case3_prepared_input_bundle,
    normalize_author_hypotheses,
)
from chemistry_multiagent.investigation.research_material_store import (
    ResearchMaterialStore,
)
from chemistry_multiagent.investigation.prompts import (
    INVESTIGATION_COMPACT_PROMPT,
    INVESTIGATION_COMPACT_PROMPT_SHA256,
    INVESTIGATION_MODEL_CONTEXT_WINDOW,
)
from chemistry_multiagent.controllers.investigation_v2_confirmation import (
    ResearchMaterialUpload,
)
from chemistry_multiagent.investigation.domain import (
    classify_investigation_failure_kind,
)
from chemistry_multiagent.utils.task_artifacts import (
    write_content_addressed_json,
    write_named_json,
)
from chemistry_multiagent.utils.llm_api import DEFAULT_MODEL, normalize_configured_model
from chemistry_multiagent.utils.expert_model_provider import (
    EXPERT_RUNTIME_STARTUP_FAILED,
    EXPERT_RUNTIME_7B,
    EXPERT_RUNTIME_CLUSTER,
    EXPERT_RUNTIME_FALLBACK,
    ExpertRuntimeSelection,
    ExpertProviderBundle,
    build_expert_provider_bundle,
    model_identity_matches,
    select_expert_runtime,
)
from chemistry_multiagent.utils.model_run_outputs import ModelRunOutputWriter
from chemistry_multiagent.utils.model_run_retrieval import RetrievalModelOutputRecorder
from chemistry_multiagent.utils.run_bundle import (
    RunBundle,
)
from chemistry_multiagent.utils.gaussian_routing import (
    GaussianRoutePolicy,
    derive_gaussian_route_policy,
    verify_frozen_gaussian_route_policy,
)
from chemistry_multiagent.utils.scientific_input import (
    merge_supplementary_chemistry_context,
    normalize_supplementary_scientific_input,
)


project_root = os.path.dirname(os.path.abspath(__file__))
# Working Memory 的任务相对路径统一从这个部署根解析。MiniChem 自己管理每个
# Action 的运行环境；Controller 这里只保存 ARCHE 的任务事实和结果文件。
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MINICHEM_CATALOG_ENDPOINT = "http://127.0.0.1:9011/mcp"
_DEFAULT_MINICHEM_INVESTIGATION_ENDPOINT = "http://127.0.0.1:9012/mcp"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _expert_role_backend_fingerprints(
    runtime: ExpertProviderBundle | None,
) -> dict[str, str | None] | None:
    """提取专家职责和通用职责的冻结 provider 配置指纹。

    输入是任务启动时冻结的 provider bundle；动作只读取角色配置指纹，不访问消息
    或密钥。返回值是 ``expert``/``general`` 指纹映射，无法读取 bundle 时返回
    ``None``；本地、可选集群或 Flash 被选中时，``expert`` 指向实际专家 provider，
    ``general`` 指向 General opinion/Judge 使用的 Pro provider。
    """

    if not isinstance(runtime, ExpertProviderBundle):
        return None
    selection = runtime.runtime_selection
    backend = (
        selection.get("backend")
        if isinstance(selection, Mapping)
        else getattr(selection, "backend", None)
    )
    if backend == EXPERT_RUNTIME_FALLBACK:
        # 回退后 ExpertModelAgent 的 expert_provider 使用独立 DS Flash；General
        # opinion/Judge 仍使用 DS Pro，两个实际调用角色必须保留不同指纹。
        return {
            "expert": runtime.fallback.config_fingerprint,
            "general": runtime.general.config_fingerprint,
        }
    if backend == EXPERT_RUNTIME_CLUSTER and runtime.cluster is not None:
        return {
            "expert": runtime.cluster.config_fingerprint,
            "general": runtime.general.config_fingerprint,
        }
    return {
        "expert": runtime.expert.config_fingerprint,
        "general": runtime.general.config_fingerprint,
    }


def _select_shared_api_key_for_expert_role(
    shared_api_key: str | None,
    *,
    explicit_role_api_key_env: str | None,
    role_environment_prefix: str,
) -> str | None:
    """返回统一的显式密钥。

    ``ARCHE_CHEM_API_KEY`` 是唯一的环境密钥入口。保留旧参数只是为了兼容调用方
    的函数签名；角色前缀和角色专用环境变量不再参与路由。
    """

    if not isinstance(role_environment_prefix, str) or not role_environment_prefix:
        raise ValueError("role_environment_prefix 必须是非空字符串")
    return shared_api_key


def _normalize_hypothesis_comparison_workers(value: int | str) -> int:
    """校验假设比较并发并返回内部整数上限。

    输入是 Controller 构造参数；动作只做类型和范围校验，不启动线程。``0``
    和 ``auto`` 都表示交给当前任务数量决定全并发，正整数表示并发上限，非法
    类型或负数立即抛出 ``ValueError``。
    """

    if isinstance(value, bool):
        raise ValueError("hypothesis_comparison_workers 不能是布尔值")
    if isinstance(value, str):
        if value.casefold() != "auto":
            raise ValueError(
                "hypothesis_comparison_workers 只能是正整数、0 或 auto"
            )
        return 0
    if isinstance(value, int):
        if value < 0:
            raise ValueError("hypothesis_comparison_workers 不能是负数")
        return value
    raise ValueError("hypothesis_comparison_workers 必须是正整数、0 或 auto")


def _normalize_hypothesis_generation_workers(value: int) -> int:
    """校验 Query 假设生成并发上限并返回正整数。

    输入是 Controller 构造参数；动作只做本地类型和范围校验，不创建线程或调用
    模型。返回值用于独立装配 HypothesisAgent；布尔值、零、负数和非整数立即
    抛出 ``ValueError``，避免把生成并发误当成比较并发。
    """

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("hypothesis_generation_workers 必须是正整数")
    return value


# ==================== 默认任务配置 ====================

_DEFAULT_TASK_CONFIG = {
    "num_queries": 3,
    "num_hypotheses_per_query": 3,
    "top_n_workflows": 5,
    "max_parallel_combinations": 5,
    "max_workflow_revisions_per_combination": 3,
    "max_hypothesis_batches": 3,
}

_DEFAULT_INVESTIGATION_TASK_CONFIG = {
    "num_queries": 3,
    "num_hypotheses_per_query": 3,
    "top_n_hypotheses": 5,
    "max_parallel_investigations": 5,
    "max_investigation_repairs_per_combination": 3,
    "max_hypothesis_batches": 3,
}


def _parse_active_investigation_endpoint(value: Any, label: str) -> str:
    """解析 active v2 的单个 MiniChem HTTP endpoint。

    输入是环境变量值和字段名；动作只做本地 URL 合同校验，不发起网络请求。返回
    去掉末尾斜杠的 ``http://loopback:port/mcp``；HTTPS、userinfo、legacy 9010、
    非 loopback、query/fragment、非法路径或端口都立即抛出 ``ValueError``。
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} 必须是非空 MCP URL")
    parsed = urlparse(value)
    if parsed.scheme != "http" or parsed.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise ValueError(f"{label} 必须是 loopback HTTP MCP URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{label} 不得包含 URL 用户名或密码")
    if parsed.path.rstrip("/") != "/mcp" or parsed.query or parsed.fragment:
        raise ValueError(f"{label} 路径必须是 /mcp 且不能带 query/fragment")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{label} 端口非法") from exc
    if port is None or not 1 <= port <= 65535:
        raise ValueError(f"{label} 必须包含合法端口")
    if port == 9010:
        raise ValueError(f"{label} 不得使用 legacy MiniChem 9010 端口")
    return value.rstrip("/")


def _validate_investigation_service_endpoints() -> tuple[str, str]:
    """检查 active v2 的目录和正式 Investigation 服务是否分离。

    输入来自环境变量；动作只解析两个 MCP URL，不发起外部请求。缺省值分别绑定
    catalog bootstrap 9011 和正式 Investigation 9012；两个 profile 使用同一地址、
    legacy 9010、userinfo、非 loopback HTTP 地址或非 ``/mcp`` 路径时立即失败，避免 Controller 启动后才把错误服务当作
    科学工具入口。返回值按 ``(catalog_endpoint, investigation_endpoint)`` 排列。
    """

    catalog_endpoint = os.environ.setdefault(
        "MINICHEM_CATALOG_ENDPOINT",
        _DEFAULT_MINICHEM_CATALOG_ENDPOINT,
    )
    investigation_endpoint = os.environ.setdefault(
        "MINICHEM_INVESTIGATION_ENDPOINT",
        _DEFAULT_MINICHEM_INVESTIGATION_ENDPOINT,
    )

    catalog_endpoint = _parse_active_investigation_endpoint(
        catalog_endpoint,
        "MINICHEM_CATALOG_ENDPOINT",
    )
    investigation_endpoint = _parse_active_investigation_endpoint(
        investigation_endpoint,
        "MINICHEM_INVESTIGATION_ENDPOINT",
    )
    if catalog_endpoint == investigation_endpoint:
        raise ValueError("catalog bootstrap 与正式 Investigation endpoint 必须分离")
    os.environ["MINICHEM_CATALOG_ENDPOINT"] = catalog_endpoint
    os.environ["MINICHEM_INVESTIGATION_ENDPOINT"] = investigation_endpoint
    return catalog_endpoint, investigation_endpoint


def _validate_expert_runtime_inputs(
    expert_model_runtime: Any,
    expert_model_provider: ExpertProvider | None,
    expert_model_expert_provider: ExpertProvider | None,
    expert_model_general_provider: ExpertProvider | None,
) -> None:
    """校验专家 bundle 与 provider 注入边界。

    输入是初始化时的候选 bundle 和三个兼容 provider 参数；动作不创建模型或发起
    探测，只拒绝已选后端、裸 provider 和错误类型。成功无返回值，失败抛出
    TypeError 或 ValueError，保证后续配置只使用可冻结候选。
    """

    if expert_model_runtime is not None and not isinstance(
        expert_model_runtime, ExpertProviderBundle
    ):
        raise TypeError("expert_model_runtime 必须是 ExpertProviderBundle 或 None")
    if (
        isinstance(expert_model_runtime, ExpertProviderBundle)
        and expert_model_runtime.runtime_selection is not None
    ):
        raise ValueError(
            "Controller 只接受尚未选择后端的候选 ExpertProviderBundle；"
            "已有任务必须从 ResearchTaskCreated 恢复冻结选择"
        )
    injected_expert_providers = (
        expert_model_provider,
        expert_model_expert_provider,
        expert_model_general_provider,
    )
    if expert_model_runtime is not None and any(
        provider is not None for provider in injected_expert_providers
    ):
        raise ValueError("expert_model_runtime 不能与裸专家 provider 同时提供")
    if any(provider is not None for provider in injected_expert_providers):
        raise ValueError(
            "Controller 的专家 provider 必须通过 ExpertProviderBundle 提供；"
            "裸 provider 只允许直接注入 ExpertModelAgent 做隔离测试"
        )


def _configure_expert_model_runtime(
    controller: Any,
    *,
    deepseek_api_key: str | None,
    expert_model_enabled: bool,
    expert_model_provider: ExpertProvider | None,
    expert_model_expert_provider: ExpertProvider | None,
    expert_model_general_provider: ExpertProvider | None,
    expert_model_runtime: ExpertProviderBundle | None,
) -> None:
    """校验并保存专家模型候选配置，不在 Controller 构造时探测 provider。

    输入是 Controller 初始化参数；动作拒绝裸 provider、已选后端或非法 bundle，
    缺省时构造候选 bundle，并创建 ExpertModelAgent。返回值无，配置写入
    expert_model 字段；任何本地合同错误立即抛出，provider 可用性由实际模型阶段检查。
    """

    _validate_expert_runtime_inputs(
        expert_model_runtime,
        expert_model_provider,
        expert_model_expert_provider,
        expert_model_general_provider,
    )
    if expert_model_runtime is None:
        # 直接 Python 入口与 CLI 使用同一默认 enabled 语义。隔离测试若要直接
        # 注入 fake provider，应直接构造 ExpertModelAgent；当前 V2 这里只保存候选，
        # 不发送启动探测请求。
        fallback_api_key = deepseek_api_key or os.environ.get("ARCHE_CHEM_API_KEY")
        general_api_key = deepseek_api_key or os.environ.get("ARCHE_CHEM_API_KEY")
        expert_model_runtime = build_expert_provider_bundle(
            enabled=expert_model_enabled,
            fallback_api_key=fallback_api_key,
            general_api_key=general_api_key,
        )
    controller.expert_model_runtime = expert_model_runtime
    controller.expert_model_candidate_runtime = (
        replace(expert_model_runtime, runtime_selection=None)
        if isinstance(expert_model_runtime, ExpertProviderBundle)
        else None
    )
    if expert_model_runtime is not None:
        expert_model_enabled = expert_model_runtime.enabled
        expert_model_expert_provider = (
            expert_model_runtime.selected_provider
            or expert_model_runtime.expert_provider
        )
        expert_model_general_provider = expert_model_runtime.general_provider
    controller.expert_model_config = ExpertModelConfig(
        enabled=expert_model_enabled,
        expert_provider=expert_model_expert_provider,
        general_provider=expert_model_general_provider,
        provider=expert_model_provider,
        runtime_selection=(
            expert_model_runtime.runtime_selection
            if expert_model_runtime is not None
            else None
        ),
        role_backend_fingerprints=_expert_role_backend_fingerprints(
            expert_model_runtime
        ),
    )
    controller.expert_model_agent = ExpertModelAgent(controller.expert_model_config)


# ==================== 主控制器 ====================

class ResearchTaskApplication:
    """按明确顺序编排检索、假设、调查实验、审查和发布。"""

    # ==================== 初始化与依赖装配 ====================

    def __init__(
        self,
        deepseek_api_key: Optional[str] = None,
        work_dir: Optional[str] = None,
        memory_enabled: bool = True,
        memory_db_path: Optional[str] = None,
        retrieval_embedder_name: str = "bge",
        hypothesis_comparison_workers: int | str = "auto",
        hypothesis_generation_workers: int = 1,
        expert_model_enabled: bool = True,
        expert_model_provider: ExpertProvider | None = None,
        expert_model_expert_provider: ExpertProvider | None = None,
        expert_model_general_provider: ExpertProvider | None = None,
        expert_model_runtime: ExpertProviderBundle | None = None,
        case_profile: str | None = None,
        use_case1_author_main_hypothesis: bool = False,
        use_case3_author_main_hypothesis: bool = False,
        use_case3_author_main_hypothesis_advisory: bool = False,
        use_case3_author_main_hypothesis_exploration: bool = False,
        v2_module_adapters: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """保存运行配置，并按冻结模式装配 Working Memory 与阶段依赖。"""

        if not isinstance(use_case1_author_main_hypothesis, bool):
            raise TypeError("use_case1_author_main_hypothesis 必须是 bool")
        if not isinstance(use_case3_author_main_hypothesis, bool):
            raise TypeError("use_case3_author_main_hypothesis 必须是 bool")
        if not isinstance(use_case3_author_main_hypothesis_advisory, bool):
            raise TypeError("use_case3_author_main_hypothesis_advisory 必须是 bool")
        if not isinstance(use_case3_author_main_hypothesis_exploration, bool):
            raise TypeError("use_case3_author_main_hypothesis_exploration 必须是 bool")
        if sum(
            bool(value)
            for value in (
                use_case1_author_main_hypothesis,
                use_case3_author_main_hypothesis,
                use_case3_author_main_hypothesis_advisory,
                use_case3_author_main_hypothesis_exploration,
            )
        ) > 1:
            raise ValueError("Case1 和 Case3 作者主假设模式不能同时启用")
        self.case_profile = validate_case_profile(case_profile)
        self._author_main_hypothesis_mode = (
            use_case1_author_main_hypothesis
            or use_case3_author_main_hypothesis
            or use_case3_author_main_hypothesis_advisory
            or use_case3_author_main_hypothesis_exploration
        )
        self._author_main_hypothesis_case_mode = (
            CASE1_AUTHOR_MAIN_HYPOTHESIS_MODE
            if use_case1_author_main_hypothesis
            else CASE3_AUTHOR_MAIN_HYPOTHESIS_MODE
            if use_case3_author_main_hypothesis
            else CASE3_AUTHOR_MAIN_HYPOTHESIS_ADVISORY_MODE
            if use_case3_author_main_hypothesis_advisory
            else CASE3_AUTHOR_MAIN_HYPOTHESIS_EXPLORATION_MODE
            if use_case3_author_main_hypothesis_exploration
            else None
        )

        self.work_dir = os.path.abspath(work_dir or project_root)
        # 新任务固定写入仓库（或明确指定的 ARCHE_RUN_STORAGE_ROOT）下的
        # outputs/research_tasks；
        # --work-dir 仅作为输入与工作目录参数保留，不改变运行 Bundle 根目录。
        run_storage_root = Path(
            os.environ.get("ARCHE_RUN_STORAGE_ROOT", str(_REPOSITORY_ROOT))
        ).expanduser().resolve(strict=False)
        forbidden_output_roots = (
            (_REPOSITORY_ROOT / "third_party").resolve(strict=False),
            (_REPOSITORY_ROOT / "references" / "src_old").resolve(strict=False),
        )
        if any(
            run_storage_root == forbidden or forbidden in run_storage_root.parents
            for forbidden in forbidden_output_roots
        ):
            raise ValueError("ARCHE_RUN_STORAGE_ROOT 不得位于第三方或旧源码目录内")
        self._run_storage_root = run_storage_root
        if deepseek_api_key:
            os.environ["ARCHE_CHEM_API_KEY"] = deepseek_api_key
        self.deepseek_api_key = deepseek_api_key or os.environ.get(
            "ARCHE_CHEM_API_KEY",
            "",
        )
        self.memory_enabled = bool(memory_enabled)
        self.memory_db_path = memory_db_path
        self._memory_db_explicit = memory_db_path is not None
        self.retrieval_embedder_name = retrieval_embedder_name
        _configure_expert_model_runtime(
            self,
            deepseek_api_key=deepseek_api_key,
            expert_model_enabled=expert_model_enabled,
            expert_model_provider=expert_model_provider,
            expert_model_expert_provider=expert_model_expert_provider,
            expert_model_general_provider=expert_model_general_provider,
            expert_model_runtime=expert_model_runtime,
        )
        self.hypothesis_comparison_workers = _normalize_hypothesis_comparison_workers(
            hypothesis_comparison_workers
        )
        self.hypothesis_generation_workers = _normalize_hypothesis_generation_workers(
            hypothesis_generation_workers
        )
        if v2_module_adapters is not None and not isinstance(
            v2_module_adapters, Mapping
        ):
            raise TypeError("v2_module_adapters 必须是 mapping")
        # 这是公开 V2 模块边界的窄替代实现接缝。生产调用不提供它，而是由
        # ResearchTaskProductionAssembly 装配真实 adapter；测试只在这里替换
        # 模块，不会把 Harness 私有 thread/turn/job 身份泄露到 Controller。
        self.v2_module_adapters = (
            None if v2_module_adapters is None else dict(v2_module_adapters)
        )

        self.working_memory_service = None
        self.memory_status = "disabled"
        if self.memory_enabled:
            if self.memory_db_path:
                try:
                    # 显式数据库路径保留调用方指定位置；默认路径在创建 run 后
                    # 绑定到该 Bundle 的 internal/state。
                    self.working_memory_service = ResearchTaskWorkingMemory(
                        self.memory_db_path,
                        enabled=True,
                        scratch_root=self._run_storage_root,
                    )
                    self.memory_db_path = self.working_memory_service.db_path
                    self.memory_status = "available"
                except Exception as exc:
                    self.memory_status = "degraded"
                    logger.warning("Working Memory 初始化失败: %s", exc)
            else:
                # 延迟到任务身份分配后创建，数据库才能落到当前 Bundle。
                self.memory_status = "available"

        self.retrieval_agent = None
        self.hypothesis_agent = None
        if not self._author_main_hypothesis_mode:
            self._initialize_agents()

        logger.info("计算化学多智能体系统控制器初始化完成")
        logger.info("工作目录: %s", self.work_dir)

    # ==================== 公开工作流入口 ====================

    def create_standard_research_task(
        self,
        scientific_question: str,
        num_queries: int = 3,
        num_hypotheses_per_query: int = 3,
        top_n_hypotheses: int = 5,
        pdf_dir: str = "papers",
        index_dir: str = "index",
        search_papers: bool = False,
        paper_search: Optional[Mapping[str, Any]] = None,
        max_parallel_investigations: int = 5,
        max_investigation_repairs_per_combination: Optional[int] = 3,
        max_hypothesis_batches: Optional[int] = _DEFAULT_INVESTIGATION_TASK_CONFIG[
            "max_hypothesis_batches"
        ],
        supplementary_scientific_input: Optional[Mapping[str, Any]] = None,
        use_case1_author_main_hypothesis: bool = False,
        use_case3_author_main_hypothesis: bool = False,
        use_case3_author_main_hypothesis_advisory: bool = False,
        use_case3_author_main_hypothesis_exploration: bool = False,
        case3_prepared_input_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """创建标准 research_task 并推进唯一的 investigation_v2 生产链。

        输入：``scientific_question`` 是非空研究问题；其余参数指定检索目录、是否
        搜索论文、查询和候选数量，以及工作流修订和候选批次的上限。显式开启
        ``use_case1_author_main_hypothesis`` 时，只允许 Case 1 固定作者夹具的单
        假设实验分支，模式会写入 ResearchTaskCreated 并由恢复事实继续。

        阶段：先校验输入并冻结任务预算，再完成 Retrieval、Hypothesis batch、
        Investigation and Experiment、任务级 Review 路由和发布。Memory On 的新任务
        固定创建 ``investigation_v2``；不会读取 ``ARCHE_PLANNER_DRIVER``，也不会把
        新任务装配到旧 Planner/Executor 闭环。Memory Off 保留为明确的未发布实验入口，
        当前合同不允许它静默退回旧架构。

        副作用：调用检索、假设、Investigation harness、Review 模型和化学工具；开启 Memory 会写入任务、
        阶段事件、执行检查点和最终结果。Memory Off 当前在进入任何阶段前明确拒绝，
        因而不会产生旧架构的工作目录结果。

        返回：返回包含 ``status``、任务或诊断身份、阶段摘要、组合结果和输出文件位置的
        字典；暂停状态还包含可恢复任务所需的说明。

        失败：空问题、非法数量或路径立即抛出 ``ValueError``；Memory Off 或缺少 v2
        Working Memory 合同会明确抛出 ``RuntimeError``；模型、工具和持久化失败由
        Investigation 入口返回可恢复结果或抛出带固定原因的运行时错误。
        """

        if not isinstance(use_case1_author_main_hypothesis, bool):
            raise TypeError("use_case1_author_main_hypothesis 必须是 bool")
        if not isinstance(use_case3_author_main_hypothesis, bool):
            raise TypeError("use_case3_author_main_hypothesis 必须是 bool")
        if not isinstance(use_case3_author_main_hypothesis_advisory, bool):
            raise TypeError("use_case3_author_main_hypothesis_advisory 必须是 bool")
        if not isinstance(use_case3_author_main_hypothesis_exploration, bool):
            raise TypeError("use_case3_author_main_hypothesis_exploration 必须是 bool")
        if sum(
            bool(value)
            for value in (
                use_case1_author_main_hypothesis,
                use_case3_author_main_hypothesis,
                use_case3_author_main_hypothesis_advisory,
                use_case3_author_main_hypothesis_exploration,
            )
        ) > 1:
            raise ValueError("Case1 和 Case3 作者主假设模式不能同时启用")
        if use_case3_author_main_hypothesis or use_case3_author_main_hypothesis_advisory:
            raise ValueError(
                "Case 3 fixed/advisory 作者模式只能恢复，创建新任务请使用主假设探索模式"
            )
        if use_case3_author_main_hypothesis_exploration and not case3_prepared_input_dir:
            raise ValueError(
                "Case 3 主假设探索新任务必须提供 case3_prepared_input_dir"
            )
        if (
            case3_prepared_input_dir is not None
            and not use_case3_author_main_hypothesis_exploration
        ):
            raise ValueError(
                "case3_prepared_input_dir 只能用于 Case 3 主假设探索模式"
            )
        if not self.memory_enabled:
            raise RuntimeError(
                "Memory Off 的 Investigation and Experiment 入口尚未发布；"
                "不会静默退回旧 Planner/Executor 架构"
            )
        author_enabled = (
            use_case1_author_main_hypothesis
            or use_case3_author_main_hypothesis
            or use_case3_author_main_hypothesis_advisory
            or use_case3_author_main_hypothesis_exploration
        )
        if author_enabled != getattr(
            self,
            "_author_main_hypothesis_mode",
            False,
        ):
            raise ValueError(
                "作者主假设模式必须在 Controller 构造时冻结，不能在运行入口切换"
            )
        if not author_enabled:
            return self._create_research_task_from_entry_strategy(
                scientific_question=scientific_question,
                num_queries=num_queries,
                num_hypotheses_per_query=num_hypotheses_per_query,
                top_n_hypotheses=top_n_hypotheses,
                pdf_dir=pdf_dir,
                index_dir=index_dir,
                search_papers=search_papers,
                paper_search=paper_search,
                max_parallel_investigations=max_parallel_investigations,
                max_investigation_repairs_per_combination=(
                    max_investigation_repairs_per_combination
                ),
                max_hypothesis_batches=max_hypothesis_batches,
                supplementary_scientific_input=supplementary_scientific_input,
            )
        return self._create_research_task_from_entry_strategy(
            scientific_question=scientific_question,
            num_queries=num_queries,
            num_hypotheses_per_query=num_hypotheses_per_query,
            top_n_hypotheses=top_n_hypotheses,
            pdf_dir=pdf_dir,
            index_dir=index_dir,
            search_papers=search_papers,
            paper_search=paper_search,
            max_parallel_investigations=max_parallel_investigations,
            max_investigation_repairs_per_combination=(
                max_investigation_repairs_per_combination
            ),
            max_hypothesis_batches=max_hypothesis_batches,
            supplementary_scientific_input=supplementary_scientific_input,
            use_case1_author_main_hypothesis=use_case1_author_main_hypothesis,
            use_case3_author_main_hypothesis=use_case3_author_main_hypothesis,
            use_case3_author_main_hypothesis_advisory=(
                use_case3_author_main_hypothesis_advisory
            ),
            use_case3_author_main_hypothesis_exploration=(
                use_case3_author_main_hypothesis_exploration
            ),
            case3_prepared_input_dir=case3_prepared_input_dir,
        )

    def create_author_hypothesis_task(
        self,
        scientific_question: str,
        author_hypotheses: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """从研究员提供的作者假设开始创建统一确认任务。

        输入是原始科学问题和 1–5 条结构化作者假设；每条假设必须包含稳定的
        ``hypothesis_ref``、``source_ref``、SHA-256 ``source_digest``、完整
        ``hypothesis`` 和 ``provenance`` mapping。动作只把这些科学输入冻结到
        ``ResearchTaskCreated``，随后进入 ``author_hypothesis_import``；材料、指导语和
        Case 配置不属于这个入口，留给确认接口。作者导入在确认边界前不探测
        9011/9012，也不创建 Investigation session 或化学作业。

        返回值是统一 V2 任务摘要；正常情况下任务会在研究员确认边界保持 ``open``。
        输入合同、Working Memory 或模块装配失败时抛出对应的运行时异常；不会回退到
        Case 专用 fixture 或旧 Planner/Executor 流程。
        """

        if not self.memory_enabled:
            raise RuntimeError(
                "Memory Off 的作者假设直入入口尚未发布；"
                "不会静默退回旧 Planner/Executor 架构"
            )
        if getattr(self, "_author_main_hypothesis_mode", False) or getattr(
            self, "case_profile", None
        ) is not None:
            raise ValueError(
                "通用作者假设直入不能与 Case 专用模式或 profile 同时使用"
            )
        normalized = normalize_author_hypotheses(author_hypotheses)
        return self._create_research_task_from_entry_strategy(
            scientific_question=scientific_question,
            author_hypotheses=normalized,
        )

    def create_top5_confirmation_task(
        self,
        scientific_question: str,
        num_queries: int = 3,
        num_hypotheses_per_query: int = 3,
        top_n_hypotheses: int = 5,
        pdf_dir: str = "papers",
        index_dir: str = "index",
        search_papers: bool = False,
        paper_search: Optional[Mapping[str, Any]] = None,
        supplementary_scientific_input: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """运行真实 Retrieval/Hypothesis 并保存五个候选后进入确认边界。

        输入是通用科学问题、结构化科学输入和论文目录；动作真实调用 Retrieval 与
        Hypothesis operation，完成查询、候选生成、关系处理、合并和排序，发布恰好五个
        Top5 候选并等待研究员确认。确认前不会初始化 MiniChem adapter/catalog、scope、
        Investigation session/composition、Action 或 Task Review；成功返回可继续的
        `open` 任务摘要，失败保留同一 Bundle/WM 供恢复。历史 v1 Top5 任务仍按原终态恢复。
        """

        return self._create_research_task_from_entry_strategy(
            # Top 5 入口仍使用标准生成策略；单批次由冻结的数量预算表达。
            scientific_question=scientific_question,
            num_queries=num_queries,
            num_hypotheses_per_query=num_hypotheses_per_query,
            top_n_hypotheses=top_n_hypotheses,
            pdf_dir=pdf_dir,
            index_dir=index_dir,
            search_papers=search_papers,
            paper_search=paper_search,
            max_parallel_investigations=5,
            max_investigation_repairs_per_combination=3,
            max_hypothesis_batches=1,
            supplementary_scientific_input=supplementary_scientific_input,
        )

    def get_hypothesis_confirmation(self, research_task_id: str) -> Dict[str, Any]:
        """只读获取 Top 5 或作者假设直入任务的当前确认视图。

        该入口不会恢复或推进任务，也不会创建 Catalog、Session、scope、模型调用
        或化学作业。任务必须已经由当前 V2 生产入口创建并停在确认阶段。
        """

        if not self.memory_enabled:
            raise RuntimeError("读取假设确认要求启用 Working Memory")
        return ResearchTaskProductionAssembly(
            self,
            module_adapters=getattr(self, "v2_module_adapters", None),
            default_module_factory=self._build_v2_production_modules,
        ).get_hypothesis_confirmation(research_task_id)

    def submit_hypothesis_confirmation_and_continue(
        self,
        research_task_id: str,
        *,
        confirmation_id: str,
        expected_research_task_sequence: int,
        decisions: Sequence[Mapping[str, Any]],
        files: Mapping[str, ResearchMaterialUpload] | None = None,
    ) -> Dict[str, Any]:
        """提交研究员选择、最终假设、可选 guidance 和材料后继续任务。

        ``decisions`` 必须完整覆盖当前候选并选择 1–5 条执行；``files`` 的键与
        decision 中的 ``upload_ref`` 对应。文件路径只由 CLI 或前端适配器读取，
        这个公开入口接收字节或二进制流，不把调用方路径保存为科学事实。
        """

        if not self.memory_enabled:
            raise RuntimeError("提交假设确认要求启用 Working Memory")
        return ResearchTaskProductionAssembly(
            self,
            module_adapters=getattr(self, "v2_module_adapters", None),
            default_module_factory=self._build_v2_production_modules,
        ).submit_hypothesis_confirmation_and_continue(
            research_task_id,
            confirmation_id=confirmation_id,
            expected_research_task_sequence=expected_research_task_sequence,
            decisions=list(decisions),
            files=files,
        )

    def _create_research_task_from_entry_strategy(
        self,
        *,
        scientific_question: str,
        num_queries: int = 3,
        num_hypotheses_per_query: int = 3,
        top_n_hypotheses: int = 5,
        pdf_dir: str = "papers",
        index_dir: str = "index",
        search_papers: bool = False,
        paper_search: Optional[Mapping[str, Any]] = None,
        max_parallel_investigations: int = 5,
        max_investigation_repairs_per_combination: Optional[int] = 3,
        max_hypothesis_batches: Optional[int] = _DEFAULT_INVESTIGATION_TASK_CONFIG[
            "max_hypothesis_batches"
        ],
        supplementary_scientific_input: Optional[Mapping[str, Any]] = None,
        author_hypotheses: Optional[Sequence[Mapping[str, Any]]] = None,
        use_case1_author_main_hypothesis: bool = False,
        use_case3_author_main_hypothesis: bool = False,
        use_case3_author_main_hypothesis_advisory: bool = False,
        use_case3_author_main_hypothesis_exploration: bool = False,
        case3_prepared_input_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """按冻结入口策略创建 research_task 并交给唯一 V2 生产装配。

        输入是标准入口或作者假设入口已经校验过的科学参数。动作只构造
        ``ResearchTaskCreated`` 所需的 canonical 任务输入、预算和策略，再调用
        ``ResearchTaskProductionAssembly``；不读取 Harness 私有目录，也不根据旧
        ``run_mode``、runtime 或旧任务身份分流。返回值是生产装配投影的
        ``research_task_id`` 摘要，装配或持久化失败直接向上抛出。
        """

        if not self.memory_enabled:
            raise RuntimeError(
                "通用生产入口要求启用 Working Memory；不会退回旧 Planner/Executor"
            )
        task_arguments = self._build_v2_production_task_arguments(
            scientific_question=scientific_question,
            num_queries=num_queries,
            num_hypotheses_per_query=num_hypotheses_per_query,
            top_n_hypotheses=top_n_hypotheses,
            pdf_dir=pdf_dir,
            index_dir=index_dir,
            search_papers=search_papers,
            paper_search=paper_search,
            max_parallel_investigations=max_parallel_investigations,
            max_investigation_repairs_per_combination=(
                max_investigation_repairs_per_combination
            ),
            max_hypothesis_batches=max_hypothesis_batches,
            supplementary_scientific_input=supplementary_scientific_input,
            author_hypotheses=author_hypotheses,
            use_case1_author_main_hypothesis=use_case1_author_main_hypothesis,
            use_case3_author_main_hypothesis=use_case3_author_main_hypothesis,
            use_case3_author_main_hypothesis_advisory=(
                use_case3_author_main_hypothesis_advisory
            ),
            use_case3_author_main_hypothesis_exploration=(
                use_case3_author_main_hypothesis_exploration
            ),
            case3_prepared_input_dir=case3_prepared_input_dir,
        )
        assembly = ResearchTaskProductionAssembly(
            self,
            module_adapters=getattr(self, "v2_module_adapters", None),
            default_module_factory=self._build_v2_production_modules,
        )
        return assembly.create_research_task(**task_arguments)

    def _build_v2_production_task_arguments(self, **kwargs: Any) -> Dict[str, Any]:
        """把公开入口参数转换成 canonical V2 任务创建映射。

        该动作只校验科学问题、检索输入、预算和作者假设来源，随后冻结任务
        ``input``、``policy`` 与 ``dependencies``。旧 ``task_id``、``run_mode``、
        ``runtime_ref`` 等名称不会进入新任务映射；历史名称只在独立迁移 reader
        中读取。作者 Case 夹具在此一次性读取并复制，恢复时不再依赖原目录。
        """

        values = dict(kwargs)
        scientific_question = values.pop("scientific_question", None)
        if not isinstance(scientific_question, str) or not scientific_question.strip():
            raise ValueError("scientific_question 必须是非空字符串")
        scientific_question = scientific_question.strip()

        num_queries = values.pop("num_queries", 3)
        num_hypotheses_per_query = values.pop("num_hypotheses_per_query", 3)
        top_n_hypotheses = values.pop("top_n_hypotheses", 5)
        max_parallel_investigations = values.pop("max_parallel_investigations", 5)
        max_repairs = values.pop(
            "max_investigation_repairs_per_combination", 3
        )
        max_batches = values.pop("max_hypothesis_batches", 3)
        raw_author_hypotheses = values.pop("author_hypotheses", None)
        author_hypotheses = (
            normalize_author_hypotheses(raw_author_hypotheses)
            if raw_author_hypotheses is not None
            else None
        )

        use_case_flags: dict[str, bool] = {}
        for name in (
            "use_case1_author_main_hypothesis",
            "use_case3_author_main_hypothesis",
            "use_case3_author_main_hypothesis_advisory",
            "use_case3_author_main_hypothesis_exploration",
        ):
            flag = values.pop(name, False)
            if not isinstance(flag, bool):
                raise TypeError(f"{name} 必须是 bool")
            use_case_flags[name] = flag
        if sum(use_case_flags.values()) > 1:
            raise ValueError("Case1 和 Case3 作者主假设模式不能同时启用")
        prepared_input_dir = values.pop("case3_prepared_input_dir", None)
        if prepared_input_dir is not None and not isinstance(
            prepared_input_dir, (str, Path)
        ):
            raise ValueError("case3_prepared_input_dir 必须是路径")

        if (
            author_hypotheses is not None
            and any(use_case_flags.values())
        ):
            raise ValueError("通用 author_hypotheses 不能与 Case 作者模式同时使用")

        pdf_dir = values.pop("pdf_dir", "papers")
        index_dir = values.pop("index_dir", "index")
        search_papers = values.pop("search_papers", False)
        paper_search = values.pop("paper_search", None)
        if not isinstance(pdf_dir, str) or not pdf_dir.strip():
            raise ValueError("pdf_dir 必须是非空字符串")
        if not isinstance(index_dir, str) or not index_dir.strip():
            raise ValueError("index_dir 必须是非空字符串")
        if not isinstance(search_papers, bool):
            raise ValueError("search_papers 必须是 bool")
        if paper_search is not None and not isinstance(paper_search, Mapping):
            raise ValueError("paper_search 必须是 mapping 或 None")
        supplementary = normalize_supplementary_scientific_input(
            values.pop("supplementary_scientific_input", None)
        )

        author_mode = (
            self._author_main_hypothesis_case_mode
            if getattr(self, "_author_main_hypothesis_mode", False)
            else None
        )
        if author_mode is not None and not any(use_case_flags.values()):
            # 构造器冻结的 Case 模式优先于动作调用，避免调用方用一个普通入口
            # 伪造另一种策略；公开入口本身已完成同样的互斥检查。
            use_case_flags = {
                **use_case_flags,
                "use_case1_author_main_hypothesis": author_mode
                == CASE1_AUTHOR_MAIN_HYPOTHESIS_MODE,
                "use_case3_author_main_hypothesis": author_mode
                == CASE3_AUTHOR_MAIN_HYPOTHESIS_MODE,
                "use_case3_author_main_hypothesis_advisory": author_mode
                == CASE3_AUTHOR_MAIN_HYPOTHESIS_ADVISORY_MODE,
                "use_case3_author_main_hypothesis_exploration": author_mode
                == CASE3_AUTHOR_MAIN_HYPOTHESIS_EXPLORATION_MODE,
            }

        fixture = None
        prepared_structure_input = None
        if author_hypotheses is not None:
            if supplementary:
                raise ValueError(
                    "通用作者假设直入不接受 supplementary_scientific_input；"
                    "材料和指导语应在确认接口提供"
                )
            if prepared_input_dir is not None:
                raise ValueError(
                    "case3_prepared_input_dir 只能用于 Case 3 主假设探索模式"
                )
        elif author_mode is not None:
            from chemistry_multiagent.investigation.author_hypothesis_import import (
                load_author_main_hypothesis,
            )

            fixture = load_author_main_hypothesis(
                scientific_question,
                author_mode=author_mode,
                supplementary_scientific_input=supplementary or None,
            )
            if author_mode == CASE3_AUTHOR_MAIN_HYPOTHESIS_EXPLORATION_MODE:
                if supplementary:
                    raise ValueError(
                        "Case 3 主假设探索模式不能携带结构化科学输入"
                    )
                if not isinstance(prepared_input_dir, (str, Path)) or not str(
                    prepared_input_dir
                ).strip():
                    raise ValueError(
                        "Case 3 主假设探索新任务必须提供 case3_prepared_input_dir"
                    )
                prepared_structure_input = load_case3_prepared_input_bundle(
                    prepared_input_dir
                )
            elif prepared_input_dir is not None:
                raise ValueError(
                    "case3_prepared_input_dir 只能用于 Case 3 主假设探索模式"
                )
            else:
                if not supplementary:
                    supplementary = {
                        "calculation_unit": copy.deepcopy(
                            dict(fixture.calculation_unit)
                        )
                    }
                    if fixture.initial_calculation_structures is not None:
                        supplementary["initial_calculation_structures"] = copy.deepcopy(
                            dict(fixture.initial_calculation_structures)
                        )
                elif fixture.initial_calculation_structures is not None and (
                    "initial_calculation_structures" not in supplementary
                ):
                    supplementary["initial_calculation_structures"] = copy.deepcopy(
                        dict(fixture.initial_calculation_structures)
                    )
                if fixture.constraint_semantics is not None:
                    supplementary["constraint_semantics"] = copy.deepcopy(
                        dict(fixture.constraint_semantics)
                    )
        elif prepared_input_dir is not None:
            raise ValueError(
                "case3_prepared_input_dir 只能用于 Case 3 主假设探索模式"
            )

        if author_mode is not None:
            task_config: Dict[str, Any] = {
                "num_queries": 1,
                "num_hypotheses_per_query": 1,
                "top_n_hypotheses": 1,
                "max_parallel_investigations": 1,
                "max_investigation_repairs_per_combination": max_repairs,
                "max_hypothesis_batches": 1,
                "investigation_combination_count": 1,
                "investigation_mode": author_mode,
            }
        else:
            task_config = {
                "num_queries": num_queries,
                "num_hypotheses_per_query": num_hypotheses_per_query,
                "top_n_hypotheses": top_n_hypotheses,
                "max_parallel_investigations": max_parallel_investigations,
                "max_investigation_repairs_per_combination": max_repairs,
                "max_hypothesis_batches": max_batches,
            }
        try:
            task_config = validate_investigation_task_config(task_config)
        except (ContractValidationError, TypeError, ValueError) as exc:
            raise ValueError("Investigation v2 task_config 非法") from exc

        entry_strategy = (
            "author_hypothesis_import"
            if author_mode is not None or author_hypotheses is not None
            else "standard_hypothesis_generation"
        )
        research_task_id = self._allocate_identity("research-task")
        task_input: Dict[str, Any] = {
            "scientific_question": scientific_question,
            "retrieval": {
                "pdf_dir": pdf_dir.strip(),
                "index_dir": index_dir.strip(),
                "search_papers": search_papers,
                "paper_search": copy.deepcopy(dict(paper_search or {})),
            },
            "task_config": copy.deepcopy(task_config),
            "supplementary_scientific_input": copy.deepcopy(dict(supplementary)),
        }
        if author_hypotheses is not None:
            task_input["author_hypotheses"] = copy.deepcopy(author_hypotheses)
        if fixture is not None:
            task_input["author_fixture"] = {
                "fixture_id": fixture.fixture_id,
                "fixture_sha256": fixture.fixture_sha256,
                "author_mode": fixture.author_mode,
                "case_profile": fixture.case_profile,
                "question_sha256": fixture.question_sha256,
                "calculation_unit_sha256": fixture.calculation_unit_sha256,
                "hypothesis": copy.deepcopy(dict(fixture.hypothesis)),
                "provenance": copy.deepcopy(dict(fixture.provenance)),
                "initial_query_angles": list(fixture.initial_query_angles),
                "retrieval_context": copy.deepcopy(dict(fixture.retrieval_context)),
                "selection": copy.deepcopy(dict(fixture.selection)),
                "calculation_unit": copy.deepcopy(dict(fixture.calculation_unit)),
                "initial_calculation_structures": copy.deepcopy(
                    dict(fixture.initial_calculation_structures or {})
                ),
            }
            if prepared_structure_input is not None:
                task_input["author_fixture"]["structure_input"] = copy.deepcopy(
                    prepared_structure_input
                )
            if fixture.constraint_semantics is not None:
                task_input["supplementary_scientific_input"][
                    "constraint_semantics"
                ] = copy.deepcopy(dict(fixture.constraint_semantics))
            if prepared_structure_input is not None:
                task_input["prepared_input"] = copy.deepcopy(prepared_structure_input)

        policy: Dict[str, Any] = {
            "stage_architecture_version": "investigation_v2",
            "entry_strategy": entry_strategy,
            "task_config": copy.deepcopy(task_config),
            "model_configuration": self._expert_model_configuration(),
        }
        if getattr(self, "case_profile", None) is not None:
            policy["case_profile"] = self.case_profile
        dependencies: Dict[str, Any] = {
            "catalog": {"status": "deferred_to_investigation_adapter"},
            "investigation_codex_thread": {
                "compact_prompt": INVESTIGATION_COMPACT_PROMPT,
                "compact_prompt_sha256": INVESTIGATION_COMPACT_PROMPT_SHA256,
                "model_context_window": INVESTIGATION_MODEL_CONTEXT_WINDOW,
            },
            "source": "research_task_application",
        }
        if values:
            raise TypeError(
                "V2 公开任务参数未消费：" + ", ".join(sorted(values))
            )
        return {
            "research_task_id": research_task_id,
            "contract_version": V2_TASK_CONTRACT_VERSION,
            "scientific_question": scientific_question,
            "task_config": task_config,
            "task_input": task_input,
            "policy": policy,
            "dependencies": dependencies,
        }

    def _build_v2_production_modules(
        self,
        *,
        task_root: Path,
        working_memory: Any,
        task_input: Mapping[str, Any],
        policy: Mapping[str, Any],
        dependencies: Mapping[str, Any],
        task_config: Mapping[str, Any],
        research_material_store: ResearchMaterialStore | None = None,
    ) -> Mapping[str, Any]:
        """装配当前生产 V2 的真实模块 adapter。"""

        from chemistry_multiagent.controllers.investigation_v2_adapters import (
            build_investigation_v2_module_adapters,
        )

        return build_investigation_v2_module_adapters(
            self,
            task_root=task_root,
            working_memory=working_memory,
            task_input=task_input,
            policy=policy,
            dependencies=dependencies,
            task_config=task_config,
            research_material_store=research_material_store,
        )

    @staticmethod
    def _allocate_identity(prefix: str) -> str:
        """生成不透明的 canonical 任务或模块边界身份。"""

        if not isinstance(prefix, str) or not prefix.strip():
            raise ValueError("identity prefix 必须是非空字符串")
        return f"{prefix.strip()}-{uuid.uuid4()}"

    def _general_model_name(self) -> str:
        """读取本次运行使用的通用模型名称，不保存认证信息。"""

        configured = getattr(self.retrieval_agent, "general_model_name", None)
        if not isinstance(configured, str) or not configured.strip():
            configured = DEFAULT_MODEL
        try:
            return normalize_configured_model(configured.strip())
        except (TypeError, ValueError):
            return DEFAULT_MODEL

    def _expert_model_configuration(self) -> Dict[str, Any]:
        """返回冻结到 ResearchTaskCreated 的专家模型公开配置。"""

        runtime = getattr(self, "expert_model_runtime", None)
        if isinstance(runtime, ExpertProviderBundle):
            metadata = copy.deepcopy(runtime.public_metadata())
            metadata.pop("runtime_selection", None)
            return metadata
        config = getattr(self, "expert_model_config", ExpertModelConfig())
        if bool(config.enabled):
            raise RuntimeError(
                "Controller 启用专家时缺少可冻结的 ExpertProviderBundle"
            )
        role = {
            "model": None,
            "base_url": None,
            "api_key_env": "",
            "api_key_configured": False,
            "config_fingerprint": "0" * 64,
        }
        return {
            "enabled": False,
            "status": "disabled",
            "expert": copy.deepcopy(role),
            "fallback": copy.deepcopy(role),
            "general": copy.deepcopy(role),
        }

    def _load_existing_model_output_writer(
        self,
        research_task_id: str,
    ) -> ModelRunOutputWriter | None:
        """只恢复已经完整存在的 research_task 展示目录。"""

        safe_research_task_id = self._safe_model_output_runtime_id(research_task_id)
        if safe_research_task_id is None:
            return None
        output_root = self._safe_model_output_root(safe_research_task_id)
        if output_root is None:
            return None
        task_path = output_root / "internal" / "audit" / "task.json"
        index_path = output_root / "internal" / "audit" / "index.json"
        if (
            output_root.is_symlink()
            or task_path.is_symlink()
            or index_path.is_symlink()
            or not task_path.is_file()
            or not index_path.is_file()
        ):
            return None
        try:
            return ModelRunOutputWriter(
                getattr(self, "_run_storage_root", self.work_dir),
                safe_research_task_id,
                persistence_mode="memory_on",
                bundle_root=output_root,
                resume=True,
            )
        except Exception:  # pragma: no cover - 展示旁路不能阻塞恢复
            logger.error("模型输出展示恢复初始化失败")
            return None

    @staticmethod
    def _safe_model_output_runtime_id(candidate: Any) -> str | None:
        """验证展示目录身份，只允许单层安全的 research_task_id。"""

        allowed_id_characters = frozenset("._-")
        if (
            not isinstance(candidate, str)
            or not candidate
            or not candidate[0].isascii()
            or not candidate[0].isalnum()
            or not all(
                character.isascii()
                and (character.isalnum() or character in allowed_id_characters)
                for character in candidate
            )
        ):
            return None
        return candidate

    def _safe_model_output_root(self, research_task_id: str) -> Path | None:
        """解析 research_task 展示目录并拒绝符号链接或越出存储根。"""

        work_root = Path(
            getattr(self, "_run_storage_root", getattr(self, "work_dir", "."))
        ).expanduser().resolve(strict=False)
        outputs_root = work_root / "outputs"
        research_tasks_root = outputs_root / "research_tasks"
        for parent in (outputs_root, research_tasks_root):
            if parent.is_symlink():
                return None
            if parent.exists() and not parent.is_dir():
                return None
            try:
                parent.resolve(strict=False).relative_to(work_root)
            except ValueError:
                return None
        output_root = research_tasks_root / research_task_id
        if output_root.is_symlink():
            return None
        try:
            output_root.resolve(strict=False).relative_to(work_root)
        except ValueError:
            return None
        return output_root

    def _read_model_output_summary(self, research_task_id: Any) -> Dict[str, Any]:
        """只读已有 research_task 展示文件，缺失时返回固定降级摘要。"""

        safe_research_task_id = self._safe_model_output_runtime_id(research_task_id)
        if safe_research_task_id is None:
            safe_research_task_id = "unknown"
        output_root = self._safe_model_output_root(safe_research_task_id)
        task_path = (
            output_root / "internal" / "audit" / "task.json"
            if output_root is not None
            else None
        )
        index_path = (
            output_root / "internal" / "audit" / "index.json"
            if output_root is not None
            else None
        )
        work_root = Path(
            getattr(self, "_run_storage_root", getattr(self, "work_dir", "."))
        ).expanduser().resolve(strict=False)
        summary: Dict[str, Any] = {
            "status": "degraded",
            "directory": str(
                output_root
                if output_root is not None
                else work_root / "outputs" / "research_tasks" / safe_research_task_id
            ),
            "call_count": 0,
            "failed_call_count": 0,
            "index_file": "internal/audit/index.json",
        }
        if output_root is None or task_path is None or index_path is None:
            summary["error"] = {
                "failure_kind": "not_available",
                "message": "该任务没有可读取的运行展示文件",
            }
            return summary
        if (
            output_root.is_symlink()
            or task_path.is_symlink()
            or index_path.is_symlink()
            or not task_path.is_file()
            or not index_path.is_file()
        ):
            summary["error"] = {
                "failure_kind": "not_available",
                "message": "该任务没有可读取的运行展示文件",
            }
            return summary
        try:
            task = json.loads(task_path.read_text(encoding="utf-8"))
            index = json.loads(index_path.read_text(encoding="utf-8"))
            if not isinstance(task, Mapping) or not isinstance(index, Mapping):
                raise ValueError("模型输出展示文件不是 object")
            calls = index.get("calls")
            if not isinstance(calls, list):
                raise ValueError("模型输出 index 缺少 calls")
            persisted_status = task.get("model_output_status", "available")
            if persisted_status not in {"available", "degraded"}:
                raise ValueError("模型输出 task 状态非法")
            summary["status"] = persisted_status
            summary["call_count"] = len(calls)
            summary["failed_call_count"] = sum(
                1
                for call in calls
                if isinstance(call, Mapping) and call.get("status") == "failed"
            )
            if persisted_status == "degraded":
                summary["error"] = {
                    "failure_kind": "write_failed",
                    "message": "模型输出展示文件写入失败",
                }
            return summary
        except Exception:  # pragma: no cover - 损坏展示文件旁路
            logger.error("模型输出展示摘要读取失败")
            summary["error"] = {
                "failure_kind": "read_failed",
                "message": "模型输出展示文件读取失败",
            }
            return summary

    def resume_research_task(
        self,
        research_task_id: str,
    ) -> Dict[str, Any]:
        """从规范 V2 ledger 恢复同一个 research_task。

        输入是原 research_task 身份；动作只装配 canonical V2 Working Memory 和模块
        Controller，并从已保存的 Task Facts 继续。历史 schema-v10 或 waiting 迁移
        必须由显式只读迁移工具处理，不在公开应用入口中分流或写入。
        """

        if not self.memory_enabled:
            raise RuntimeError("resume_research_task 要求已启用 Working Memory")
        if not isinstance(research_task_id, str) or not research_task_id.strip():
            raise ValueError("research_task_id 必须是非空字符串")
        _validate_investigation_service_endpoints()
        return ResearchTaskProductionAssembly(
            self,
            module_adapters=getattr(self, "v2_module_adapters", None),
            default_module_factory=self._build_v2_production_modules,
        ).resume_research_task(research_task_id.strip())

    def _attach_readonly_model_output_summary(
        self,
        result: Mapping[str, Any],
        runtime_id: Any,
    ) -> Dict[str, Any]:
        """给暂停或终态结果附加展示摘要，并同步已有 Bundle 状态。"""

        projected = copy.deepcopy(dict(result))
        self._synchronize_readonly_bundle_result(projected, runtime_id)
        projected["model_outputs"] = self._read_model_output_summary(runtime_id)
        return projected

    def _result_recovery_for_bundle(
        self,
        result: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        """读取 V2 结果已经提交的恢复视图。

        V2 的恢复身份由 Investigation harness 自己写入结果；Controller 只把已
        提交的视图投影到 Run Bundle，不再根据旧 ``waiting_execution`` 或暂停字段
        猜测恢复信息。缺少显式视图时返回 ``None``，保持展示状态而不制造旧协议。
        """

        recovery = result.get("recovery")
        return recovery if isinstance(recovery, Mapping) else None

    def _synchronize_readonly_bundle_result(
        self,
        result: Mapping[str, Any],
        runtime_id: Any,
        *,
        strict: bool = False,
    ) -> RunBundle | None:
        """在 writer 缺失时把结果写回同一 Bundle。

        输入是任务结果、Bundle 身份和可选的严格标记；动作写入状态、总摘要及可选的
        ``final_result`` 阶段。普通展示旁路继续吞掉写入失败并记录固定日志；严格终态
        调用要求异常向上传播，供生产失败收束显式失败，避免返回成功但 Bundle 仍为
        ``running``。成功返回同一 Bundle 对象，身份非法或普通旁路写入失败返回
        ``None``；不读取模型正文，也不修改 Working Memory。
        """

        safe_runtime_id = self._safe_model_output_runtime_id(runtime_id)
        status = result.get("status")
        if safe_runtime_id is None or not isinstance(status, str):
            return None
        try:
            bundle = RunBundle.open_existing_research_task(
                self._run_storage_root,
                safe_runtime_id,
            )
            recovery = self._result_recovery_for_bundle(result)
            bundle.update_status(
                status,
                reason_code=(
                    result.get("reason_code")
                    if isinstance(result.get("reason_code"), str)
                    else None
                ),
                blocking=(
                    result.get("blocking")
                    if isinstance(result.get("blocking"), Mapping)
                    else None
                ),
                recovery=recovery,
            )
            bundle_result = dict(result)
            if recovery is not None:
                bundle_result["recovery"] = recovery
            bundle.update_summary(result=bundle_result)
            if status in {
                "succeeded",
                "failed",
                "stopped_no_supported_hypothesis",
            }:
                bundle.update_stage_status(
                    "final_result",
                    "completed",
                    summary=bundle_result,
                )
            return bundle
        except Exception as exc:  # pragma: no cover - 普通展示旁路不能阻塞科学结果
            logger.error("Run Bundle 只读结果状态同步失败")
            if strict:
                raise RuntimeError("Run Bundle 终态状态同步失败") from exc
            return None

    def _attach_model_output_summary(
        self,
        result: Mapping[str, Any],
        writer: ModelRunOutputWriter | None,
    ) -> Dict[str, Any]:
        """在不改变科学结果的前提下更新展示生命周期并附加摘要。

        ``writer`` 只写项目目录中的可读副本，不能参与 Working Memory 或进程内状态
        的提交。展示写入失败会被转换成 ``degraded`` 摘要，原始 status、结果文件和
        命令行退出码都保持不变。
        """

        projected = copy.deepcopy(dict(result))
        if writer is None:
            runtime_id = self._safe_model_output_runtime_id(
                projected.get("research_task_id")
            )
            if runtime_id is not None and isinstance(projected.get("status"), str):
                self._synchronize_readonly_bundle_result(projected, runtime_id)
            projected["model_outputs"] = self._build_degraded_model_output_summary(
                projected
            )
            return projected
        try:
            status = projected.get("status")
            if status in {
                "running",
                "retrieval_incomplete",
                "paused",
                "succeeded",
                "failed",
                "stopped_no_supported_hypothesis",
            }:
                writer.update_task_status(status, projected.get("reason_code"))
            writer.update_run_bundle(projected)
            projected["model_outputs"] = writer.result_summary()
        except Exception:  # pragma: no cover - writer 已承诺隔离 I/O 失败
            logger.error("模型输出展示生命周期更新失败")
            try:
                projected["model_outputs"] = writer.result_summary()
            except Exception:
                projected["model_outputs"] = (
                    self._build_degraded_model_output_summary(projected)
                )
        return projected

    def _build_degraded_model_output_summary(
        self,
        result: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """构造不写文件的模型展示降级摘要。

        输入是一次已经完成的科学结果。方法只读取结果中的任务身份，并用模型输出
        writer 相同的身份规则防止 ``research_task_id`` 把展示目录带出工作目录；身份非法或
        缺失时使用固定的 ``unknown`` 目录名。返回的摘要不会创建目录、修改任务状态
        或影响命令行退出码。
        """

        runtime_id = self._safe_model_output_runtime_id(
            result.get("research_task_id")
        )
        if runtime_id is None:
            runtime_id = "unknown"
        work_dir = Path(
            getattr(self, "_run_storage_root", getattr(self, "work_dir", "."))
        ).expanduser().resolve(strict=False)
        return {
            "status": "degraded",
            "directory": str(work_dir / "outputs" / "research_tasks" / runtime_id),
            "call_count": 0,
            "failed_call_count": 0,
            "index_file": "internal/audit/index.json",
            "error": {
                "failure_kind": "write_failed",
                "message": "模型输出展示文件写入失败",
            },
        }


    def run_retrieval_phase(
        self,
        request: Mapping[str, Any],
        receipt_sink: Any = None,
        *,
        frozen_chemistry_context: Optional[Mapping[str, Any]] = None,
        frozen_chemistry_context_error: Optional[Mapping[str, Any]] = None,
        supplementary_scientific_input: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """按固定请求运行 Retrieval 阶段，并返回可冻结的科学上下文。

        输入必须包含科学问题、论文目录、索引目录和搜索开关。方法调用 Retrieval
        Agent，可选地把内部收据交给调用方保存；不会创建假设批次或修改 Working
        Memory。Agent 未初始化、请求非法或返回值不是 JSON object 时直接抛错。
        """

        required_fields = {
            "scientific_question",
            "pdf_dir",
            "index_dir",
            "search_papers",
        }
        if not isinstance(request, Mapping) or set(request) not in (
            required_fields,
            required_fields | {"paper_search"},
        ):
            raise ValueError("Retrieval request 字段不符合固定形状")
        question = request["scientific_question"]
        if (
            not isinstance(question, str)
            or not question.strip()
            or not isinstance(request["pdf_dir"], str)
            or not request["pdf_dir"]
            or not isinstance(request["index_dir"], str)
            or not request["index_dir"]
            or not isinstance(request["search_papers"], bool)
        ):
            raise ValueError("Retrieval request 字段值非法")
        if self.retrieval_agent is None:
            raise RuntimeError("Retrieval Agent 未完成初始化")
        paper_search = normalize_paper_search_config(
            request.get("paper_search"),
            enabled=request["search_papers"],
        )
        normalized_supplementary_input = normalize_supplementary_scientific_input(
            supplementary_scientific_input
        )
        result = self.retrieval_agent.process_question(
            question=question,
            pdf_dir=request["pdf_dir"],
            index_dir=request["index_dir"],
            search_papers=request["search_papers"],
            paper_search=paper_search,
            frozen_chemistry_context=frozen_chemistry_context,
            frozen_chemistry_context_error=frozen_chemistry_context_error,
            supplementary_scientific_input=normalized_supplementary_input,
            receipt_sink=receipt_sink,
        )
        if not isinstance(result, dict):
            raise RuntimeError("Retrieval Agent 必须返回 JSON object")
        return result

    @staticmethod
    def _build_retrieval_pause_status(
        request: Mapping[str, Any],
        retrieval_result: Mapping[str, Any],
        *,
        reason_code: str,
    ) -> Dict[str, Any]:
        """从 Retrieval 结果整理不含绝对路径的任务级暂停事实。"""

        configured = request.get("paper_search")
        config = normalize_paper_search_config(
            configured if isinstance(configured, Mapping) else None,
            enabled=bool(request.get("search_papers", False)),
        )
        raw_status = retrieval_result.get("paper_search_status")
        source = raw_status if isinstance(raw_status, Mapping) else {}

        def path_ref(value: Any, default_name: str) -> str:
            if not isinstance(value, str) or not value.strip():
                return default_name
            candidate = Path(value.strip())
            if candidate.is_absolute():
                return candidate.name or default_name
            return PurePosixPath(value.strip()).as_posix()

        manifest_ref = path_ref(
            source.get("manifest_path") or config.get("manifest_path"),
            ".paper-search-manifest.json",
        )
        resume_ref = path_ref(
            source.get("resume_state_path") or config.get("resume_state_path"),
            ".paper-search-resume.json",
        )
        cache_ref = path_ref(
            source.get("response_cache_dir") or config.get("response_cache_dir"),
            ".paper-search-cache",
        )
        actual = source.get("actual_papers", 0)
        if isinstance(actual, bool) or not isinstance(actual, int) or actual < 0:
            actual = 0
        target = source.get("target_papers", config.get("target_papers", 50))
        if isinstance(target, bool) or not isinstance(target, int) or target <= 0:
            target = 50
        stop_reason = source.get("stop_reason")
        if not isinstance(stop_reason, str) or not stop_reason.strip():
            stop_reason = reason_code
        return {
            "status": "recoverable_incomplete",
            "operation": "paper_search",
            "reason_code": reason_code,
            "provider": str(source.get("provider") or config.get("provider_strategy")),
            "manifest_path": manifest_ref,
            "resume_state_path": resume_ref,
            "response_cache_dir": cache_ref,
            "target_papers": target,
            "actual_papers": actual,
            "target_reached": False,
            "stop_reason": stop_reason,
        }

    @staticmethod
    def _paper_search_has_shortfall(status: Any) -> bool:
        """判断 PaperSearch 是否仍有全局论文目标缺口。

        输入是 Retrieval 返回的 ``paper_search_status``。动作：同时检查显式未完成
        标记、``target_reached`` 和实际/目标数量，兼容 provider 在数量不足时返回
        ``partial`` 或 ``completed`` 的不一致状态。返回 True 表示不能进入
        Hypothesis；空值或完整目标返回 False，不把 ``chemistry_context_error`` 当作
        Retrieval 根错误。
        """

        if not isinstance(status, Mapping):
            return False
        if status.get("target_reached") is False:
            return True
        status_name = status.get("status")
        if status_name in {
            "recoverable_incomplete",
            "partial",
            "incomplete",
            "shortfall",
        }:
            return True
        actual = status.get("actual_papers")
        target = status.get("target_papers")
        if (
            isinstance(actual, int)
            and not isinstance(actual, bool)
            and isinstance(target, int)
            and not isinstance(target, bool)
            and target > 0
            and actual < target
        ):
            return True
        return False


    # ==================== Agent 装配与结果整理 ====================

    def _initialize_agents(self) -> None:
        """初始化标准 Investigation v2 所需的 Retrieval 和 Hypothesis 智能体。

        当前 V2 不创建旧 Planner、Reflection 或旧执行闭环对象；作者主假设直入
        模式在入口冻结时跳过本方法。
        """

        try:
            self.retrieval_agent = RetrievalAgent(
                deepseek_api_key=self.deepseek_api_key,
                embedder_name=self.retrieval_embedder_name,
            )
            logger.info("✅ Retrieval Agent 初始化完成")
        except Exception as exc:
            logger.error("Retrieval Agent 初始化失败: %s", exc)

        try:
            self.hypothesis_agent = HypothesisAgent(
                comparison_workers=self.hypothesis_comparison_workers,
                hypothesis_generation_workers=self.hypothesis_generation_workers,
            )
            logger.info("✅ Hypothesis Agent 初始化完成")
        except Exception as exc:
            logger.error("Hypothesis Agent 初始化失败: %s", exc)



# ==================== 命令行入口 ====================

def _select_retrieval_embedder_name(search_papers: bool) -> str:
    """选择检索嵌入后端；是否下载论文不改变本地向量编码。"""

    del search_papers
    return "bge"


def _project_schema_v10_legacy_result_identity(
    result: Mapping[str, Any],
    projected: Dict[str, Any],
) -> None:
    """在显式 schema-v10 读取边界把历史身份投影成 canonical 名称。

    输入只允许来自已完成的历史读取结果；动作不把旧字段复制到公开 JSON，也不
    参与当前生产任务的结果构造。缺少 canonical 身份时按历史形状一次性转换，无法
    证明身份时保持缺失，由调用方继续按失败或未知状态处理。
    """

    if "research_task_id" not in projected:
        for legacy_name in ("task_id", "inquiry_id"):
            if legacy_name in result:
                projected["research_task_id"] = result[legacy_name]
                break
    if "module_invocation_id" not in projected:
        legacy_name = "module_run_id"
        if legacy_name in result:
            projected["module_invocation_id"] = result[legacy_name]


def _cli_result_view(result: Mapping[str, Any]) -> Dict[str, Any]:
    """保留命令行后续操作真正需要的结果，省略体积较大的内部正文。

    普通运行结果仍以状态和任务身份为主；一旦任务暂停、查询恢复说明或应用人工
    恢复，暂停身份、允许动作和提交结果也必须显示，否则研究员无法完成下一条
    命令。这里仅复制已有返回字段，不创建第二套恢复协议。
    """

    visible_keys = (
        "status",
        "research_task_id",
        "module_invocation_id",
        "module_input_digest",
        "input_scientific_record_ids",
        "expected_research_task_sequence",
        "research_task_sequence",
        "hypothesis_batch_record_id",
        "hypothesis_candidate_record_id",
        "candidates",
        "confirmation_id",
        "memory_status",
        "output_files",
        "reason_code",
        "expert_review_stops",
        "pause_id",
        "restart_guide",
        "pause_ids",
        "restart_guides",
        "combination_results",
        "guide",
        "results",
        "model_outputs",
        "paper_search_status",
        "retrieval_result",
    )
    projected = {
        key: result[key]
        for key in visible_keys
        if key in result
    }
    _project_schema_v10_legacy_result_identity(result, projected)
    projected.setdefault("status", "unknown")
    projected.setdefault("research_task_id", None)
    return projected


def _load_strict_scientific_input_json(path_value: str) -> Dict[str, Any]:
    """读取本地 JSON 并返回已校验的计算单元副本。

    输入路径只在控制层使用，不写入返回值；函数不修改文件。路径、编码、JSON
    结构或字段合同非法时抛出 ``ValueError``。
    """

    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("scientific input JSON 路径必须是非空字符串")
    input_path = Path(path_value).expanduser()
    if not input_path.is_file() or input_path.is_symlink():
        raise ValueError("scientific input JSON 必须是普通文件")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"scientific input JSON 含重复字段：{key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            input_path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"scientific input JSON 含非法数字：{value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("scientific input JSON 无法严格解析") from exc
    return normalize_supplementary_scientific_input(payload)


def _load_strict_author_hypotheses_json(path_value: str) -> list[dict[str, Any]]:
    """读取作者假设 JSON 文件并转换为统一结构化输入。

    输入路径只在 CLI 适配层使用；动作严格解析 JSON、拒绝符号链接和重复字段，
    再调用作者假设合同校验。返回值不含路径或材料正文，失败时抛出 ``ValueError``。
    """

    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("author hypotheses JSON 路径必须是非空字符串")
    input_path = Path(path_value).expanduser()
    if not input_path.is_file() or input_path.is_symlink():
        raise ValueError("author hypotheses JSON 必须是普通文件")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"author hypotheses JSON 含重复字段：{key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            input_path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"author hypotheses JSON 含非法数字：{value}")
            ),
        )
        return normalize_author_hypotheses(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("author hypotheses JSON 无法严格解析") from exc


def _load_hypothesis_confirmation_request(
    path_value: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """严格解析 CLI 确认 JSON，并把本地文件路径留在入口适配层。

    返回的第一项只含后端确认合同字段；第二项是 CLI 私有文件描述，调用方随后
    以普通只读文件流打开。相对路径以确认 JSON 所在目录为基准。重复字段、符号
    链接、特殊文件或字段漂移均在调用 Controller 前拒绝。
    """

    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("confirmation request JSON 路径必须是非空字符串")
    request_path = Path(path_value).expanduser()
    if not request_path.is_file() or request_path.is_symlink():
        raise ValueError("confirmation request JSON 必须是普通文件")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"confirmation request JSON 含重复字段：{key}")
            result[key] = value
        return result

    try:
        raw = json.loads(
            request_path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"confirmation request JSON 含非法数字：{value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("confirmation request JSON 无法严格解析") from exc
    if not isinstance(raw, Mapping) or set(raw).difference(
        {
            "confirmation_id",
            "expected_research_task_sequence",
            "decisions",
            "files",
        }
    ) or not {
        "confirmation_id",
        "expected_research_task_sequence",
        "decisions",
    }.issubset(raw):
        raise ValueError("confirmation request JSON 字段合同不匹配")
    raw_files = raw.get("files", {})
    if not isinstance(raw_files, Mapping):
        raise ValueError("confirmation request files 必须是 object")
    file_specs: dict[str, dict[str, Any]] = {}
    request_directory = request_path.absolute().parent
    for upload_ref, value in raw_files.items():
        if not isinstance(upload_ref, str) or not upload_ref.strip():
            raise ValueError("confirmation request upload_ref 必须是非空字符串")
        if not isinstance(value, Mapping) or set(value) != {
            "path",
            "filename",
            "media_type",
        }:
            raise ValueError(
                f"confirmation request files[{upload_ref}] 字段合同不匹配"
            )
        source_path = value["path"]
        if not isinstance(source_path, str) or not source_path.strip():
            raise ValueError(
                f"confirmation request files[{upload_ref}].path 必须是非空字符串"
            )
        local_path = Path(source_path).expanduser()
        if not local_path.is_absolute():
            local_path = request_directory / local_path
        local_path = local_path.absolute()
        current = Path(local_path.anchor)
        for part in local_path.parts[1:]:
            current /= part
            try:
                metadata = os.lstat(current)
            except OSError as exc:
                raise ValueError(
                    f"confirmation request files[{upload_ref}] 无法读取"
                ) from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError("confirmation request 不接受符号链接文件")
        if not stat.S_ISREG(os.lstat(local_path).st_mode):
            raise ValueError("confirmation request 只接受普通文件")
        file_specs[upload_ref] = {
            "path": local_path,
            "filename": value["filename"],
            "media_type": value["media_type"],
        }
    return (
        {
            "confirmation_id": raw["confirmation_id"],
            "expected_research_task_sequence": raw[
                "expected_research_task_sequence"
            ],
            "decisions": raw["decisions"],
        },
        file_specs,
    )


def _build_cli_parser() -> argparse.ArgumentParser:
    """声明 research_task 六个公开动作及其参数。"""

    parser = argparse.ArgumentParser(
        description="计算化学多智能体系统 research_task 控制器"
    )
    actions = parser.add_subparsers(
        dest="action",
        required=True,
        metavar="ACTION",
    )

    def add_common_options(command: argparse.ArgumentParser) -> None:
        """为每个动作添加运行根、记忆和模型候选参数。"""

        command.add_argument("--api-key", help="模型 API 密钥，也可由环境变量提供")
        command.add_argument("--work-dir", help="工作目录")
        command.add_argument("--memory-db", help="Working Memory 数据库路径")
        command.add_argument(
            "--disable-memory",
            action="store_true",
            help="关闭 Working Memory（当前公开生产动作不接受）",
        )
        command.add_argument(
            "--disable-expert-model",
            action="store_true",
            help="显式关闭 Gaussian 专家审查和完整结果评价",
        )
        command.add_argument(
            "--expert-model",
            help="Gaussian 专家模型公开标识；省略时使用 ARCHE_CHEM_MODEL_NAME",
        )
        command.add_argument("--expert-base-url", help="Gaussian 专家模型 OpenAI-compatible 地址")
        command.add_argument(
            "--expert-api-key-env",
            default=None,
            help="Gaussian 专家模型密钥环境变量名称（默认 ARCHE_CHEM_API_KEY）",
        )
        command.add_argument(
            "--expert-fallback-model",
            help="Gaussian 专家回退模型；省略时使用 ARCHE_CHEM_MODEL_NAME",
        )
        command.add_argument(
            "--expert-fallback-base-url",
            help="Gaussian 专家回退模型 OpenAI-compatible 地址",
        )
        command.add_argument(
            "--expert-fallback-api-key-env",
            default=None,
            help="Gaussian 专家回退模型密钥所在环境变量名称（默认 ARCHE_CHEM_API_KEY）",
        )
        command.add_argument(
            "--expert-general-model",
            help="专家 debate 的通用意见/裁判模型；省略时使用 ARCHE_CHEM_MODEL_NAME",
        )
        command.add_argument(
            "--expert-general-base-url",
            help="专家 debate 通用模型 OpenAI-compatible 地址",
        )
        command.add_argument(
            "--expert-general-api-key-env",
            default=None,
            help="专家 debate 通用模型密钥所在环境变量名称（默认 ARCHE_CHEM_API_KEY）",
        )
        command.add_argument(
            "--hypothesis-comparison-workers",
            type=_parse_hypothesis_comparison_workers,
            default="auto",
            help="假设比较并发上限；0 或 auto 表示当前任务数全并发",
        )
        command.add_argument(
            "--hypothesis-generation-workers",
            type=int,
            default=1,
            help="各 Query 假设生成子调用的最大并发数",
        )
        command.add_argument(
            "--case-profile",
            choices=tuple(sorted(SUPPORTED_CASE_PROFILES)),
            default=None,
            help=argparse.SUPPRESS,
        )

    def add_paper_search_options(command: argparse.ArgumentParser) -> None:
        """为需要 Retrieval 的动作添加论文和结构化输入参数。"""

        command.add_argument("--scientific-input-json", help="结构化计算单元 JSON 文件")
        command.add_argument("--pdf-dir", default="papers", help="论文目录")
        command.add_argument("--index-dir", default="index", help="索引目录")
        command.add_argument("--no-search", action="store_true", help="跳过论文检索")
        command.add_argument("--paper-search-manifest", help="PaperSearch manifest 路径")
        command.add_argument("--paper-search-resume-state", help="PaperSearch 恢复文件路径")
        command.add_argument("--paper-search-cache-dir", help="PaperSearch Scholar 响应缓存目录")
        command.add_argument(
            "--paper-search-target-papers",
            type=int,
            default=50,
            help="PaperSearch 全局去重 PDF 目标",
        )
        command.add_argument(
            "--paper-search-timeout-seconds",
            type=float,
            default=180.0,
            help="单关键词 PaperSearch 墙钟上限",
        )
        command.add_argument(
            "--paper-search-keyword-delay-seconds",
            type=float,
            default=45.0,
            help="关键词间退避秒数",
        )
        command.add_argument(
            "--paper-search-keyword-jitter-seconds",
            type=float,
            default=45.0,
            help="关键词间随机抖动上限",
        )
        command.add_argument(
            "--paper-search-download-delay-seconds",
            type=float,
            default=1.0,
            help="PDF 下载最小间隔",
        )
        command.add_argument(
            "--paper-search-scholar-max-pages",
            type=int,
            default=3,
            help="Scholar 每关键词最大页数",
        )
        command.add_argument(
            "--paper-search-scholar-results-per-page",
            type=int,
            default=10,
            help="Scholar 每页结果数",
        )
        command.add_argument(
            "--paper-search-scholar-crossref-max-candidates",
            type=int,
            default=3,
            help="Scholar 每页 Crossref 补充上限",
        )
        command.add_argument(
            "--paper-search-scholar-request-delay-seconds",
            type=float,
            default=15.0,
            help="Scholar 页面最小间隔",
        )
        command.add_argument(
            "--paper-search-scholar-request-jitter-seconds",
            type=float,
            default=15.0,
            help="Scholar 页面随机抖动",
        )
        command.add_argument(
            "--paper-search-scholar-initial-delay-seconds",
            type=float,
            default=10.0,
            help="首次 Scholar 请求预等待",
        )
        command.add_argument(
            "--paper-search-scholar-initial-jitter-seconds",
            type=float,
            default=20.0,
            help="首次 Scholar 请求随机抖动",
        )
        command.add_argument(
            "--paper-search-scholar-backoff-base-seconds",
            type=float,
            default=900.0,
            help="Scholar 限流退避基础秒数",
        )
        command.add_argument(
            "--paper-search-scholar-max-backoff-seconds",
            type=float,
            default=3600.0,
            help="Scholar 最大冷却秒数",
        )
        command.add_argument(
            "--paper-search-scholar-max-consecutive-rate-limits",
            type=int,
            default=1,
            help="Scholar 连续限流停机次数",
        )

    standard = actions.add_parser(
        "create-standard-research-task",
        help="创建标准假设生成 research_task",
    )
    add_common_options(standard)
    standard.add_argument("--question", "-q", required=True, help="科学问题")
    add_paper_search_options(standard)
    standard.add_argument("--num-queries", type=int, default=3, help="查询数量")
    standard.add_argument(
        "--num-hypotheses-per-query",
        type=int,
        default=3,
        help="每个查询的假设数量",
    )
    standard.add_argument(
        "--top-n-hypotheses",
        type=int,
        default=5,
        help="当前批次保留的假设数量，固定为 5",
    )
    standard.add_argument(
        "--max-parallel-investigations",
        type=int,
        default=5,
        help="Investigation 同时运行的组合上限",
    )
    standard.add_argument(
        "--max-investigation-repairs-per-combination",
        type=int,
        default=3,
        help="每个组合的 Investigation repair 次数",
    )
    standard.add_argument(
        "--max-hypothesis-batches",
        type=int,
        default=_DEFAULT_INVESTIGATION_TASK_CONFIG["max_hypothesis_batches"],
        help="任务最多生成的假设批次数",
    )
    standard.add_argument(
        "--use-case1-author-main-hypothesis",
        action="store_true",
        help="内部 Case 1 实验：导入固定作者主假设并只运行一个组合",
    )
    standard.add_argument(
        "--use-case3-author-main-hypothesis",
        action="store_true",
        help="内部 Case 3 实验：导入固定作者主假设并只运行一个组合",
    )
    standard.add_argument(
        "--use-case3-author-main-hypothesis-advisory",
        action="store_true",
        help="内部 Case 3 advisory 实验：导入作者主假设并只运行一个组合",
    )
    standard.add_argument(
        "--use-case3-author-main-hypothesis-exploration",
        action="store_true",
        help="Case 3 实验：提供原题、作者主假设和清理结构，由模型自行探索",
    )
    standard.add_argument(
        "--case3-prepared-input-dir",
        help="Case 3 主假设探索使用的五文件清理输入目录",
    )

    author = actions.add_parser(
        "create-author-hypothesis-task",
        help="从作者假设创建 research_task",
    )
    add_common_options(author)
    author.add_argument("--question", "-q", required=True, help="科学问题")
    author.add_argument(
        "--author-hypotheses-json",
        required=True,
        help="作者假设 JSON 文件",
    )

    top5 = actions.add_parser(
        "create-top5-confirmation-task",
        help="创建 Top 5 确认 research_task",
    )
    add_common_options(top5)
    top5.add_argument("--question", "-q", required=True, help="科学问题")
    add_paper_search_options(top5)
    top5.add_argument("--num-queries", type=int, default=3, help="查询数量")
    top5.add_argument(
        "--num-hypotheses-per-query",
        type=int,
        default=3,
        help="每个查询的假设数量",
    )
    top5.add_argument(
        "--top-n-hypotheses",
        type=int,
        default=5,
        help="候选数量，固定为 5",
    )

    get_confirmation = actions.add_parser(
        "get-hypothesis-confirmation",
        help="读取 research_task 的假设确认视图",
    )
    add_common_options(get_confirmation)
    get_confirmation.add_argument("research_task_id")

    submit_confirmation = actions.add_parser(
        "submit-hypothesis-confirmation-and-continue",
        help="提交确认决定并继续 research_task",
    )
    add_common_options(submit_confirmation)
    submit_confirmation.add_argument("research_task_id")
    submit_confirmation.add_argument(
        "--confirmation-request-json",
        required=True,
        help="确认决定及材料描述 JSON",
    )

    resume = actions.add_parser(
        "resume-research-task",
        help="恢复已有 research_task",
    )
    add_common_options(resume)
    resume.add_argument("research_task_id")

    return parser

def _parse_hypothesis_comparison_workers(value: str) -> int | str:
    """解析假设比较并发参数，保留 0/auto 的全并发语义。"""

    if isinstance(value, str) and value.casefold() == "auto":
        return "auto"
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "假设比较并发数必须是正整数、0 或 auto"
        ) from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("假设比较并发数不能是负数")
    return parsed


def _validate_cli_action_selection(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    """校验六个公开动作的专属参数，不接受旧动作或旧运行模式。"""

    action = getattr(args, "action", None)
    supported_actions = {
        "create-standard-research-task",
        "create-author-hypothesis-task",
        "create-top5-confirmation-task",
        "get-hypothesis-confirmation",
        "submit-hypothesis-confirmation-and-continue",
        "resume-research-task",
    }
    if action not in supported_actions:
        parser.error("必须选择一个 research_task 公开动作")
    if getattr(args, "disable_memory", False):
        parser.error("research_task 公开生产动作必须启用 Working Memory")
    if action == "create-standard-research-task":
        exploration = bool(
            getattr(args, "use_case3_author_main_hypothesis_exploration", False)
        )
        case_flags = (
            bool(getattr(args, "use_case1_author_main_hypothesis", False)),
            bool(getattr(args, "use_case3_author_main_hypothesis", False)),
            bool(getattr(args, "use_case3_author_main_hypothesis_advisory", False)),
            exploration,
        )
        if sum(case_flags) > 1:
            parser.error("Case1 和 Case3 作者主假设模式不能同时启用")
        if (
            getattr(args, "use_case3_author_main_hypothesis", False)
            or getattr(args, "use_case3_author_main_hypothesis_advisory", False)
        ):
            parser.error(
                "Case 3 fixed/advisory 作者模式只能恢复，创建新任务请使用主假设探索模式"
            )
        if exploration and not getattr(args, "case3_prepared_input_dir", None):
            parser.error("Case 3 主假设探索模式必须提供 --case3-prepared-input-dir")
        if getattr(args, "case3_prepared_input_dir", None) and not exploration:
            parser.error("--case3-prepared-input-dir 只能用于 Case 3 主假设探索模式")


def _validate_investigation_cli_limits(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    """校验创建动作的固定假设数量、并发和 repair 预算。"""

    action = getattr(args, "action", None)
    if action == "create-top5-confirmation-task":
        if args.top_n_hypotheses != 5:
            parser.error("Top5 确认动作的 --top-n-hypotheses 必须固定为 5")
        return
    if action != "create-standard-research-task":
        return
    exploration = bool(
        getattr(args, "use_case3_author_main_hypothesis_exploration", False)
    )
    author_case = (
        bool(getattr(args, "use_case1_author_main_hypothesis", False))
        or bool(getattr(args, "use_case3_author_main_hypothesis", False))
        or bool(getattr(args, "use_case3_author_main_hypothesis_advisory", False))
        or exploration
    )
    if author_case:
        allowed = {
            "num_queries": {1, 3},
            "num_hypotheses_per_query": {1, 3},
            "top_n_hypotheses": {1, 5},
            "max_parallel_investigations": {1, 5},
            "max_hypothesis_batches": {1, 3},
        }
        for field, values in allowed.items():
            if getattr(args, field) not in values:
                parser.error(
                    f"作者主假设实验模式固定单组合；--{field.replace('_', '-')}"
                    " 不得覆盖为其它预算"
                )
        return
    if args.top_n_hypotheses != 5:
        parser.error("标准 research_task 的 --top-n-hypotheses 必须固定为 5")
    if not 1 <= args.max_parallel_investigations <= 5:
        parser.error("--max-parallel-investigations 必须是 1 到 5 的整数")
    repairs = args.max_investigation_repairs_per_combination
    if isinstance(repairs, bool) or not isinstance(repairs, int) or repairs < 1:
        parser.error("--max-investigation-repairs-per-combination 必须是正整数")


def _validate_cli_request(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    """校验命令行动作互斥关系。"""

    _validate_cli_action_selection(parser, args)
    _validate_investigation_cli_limits(parser, args)


def _controller_from_cli_args(
    args: argparse.Namespace,
) -> ResearchTaskApplication:
    """用已经校验的公开动作参数装配 Controller。"""

    expert_model_enabled = not bool(getattr(args, "disable_expert_model", False))
    if expert_model_enabled:
        expert_api_key_env = (
            getattr(args, "expert_api_key_env", None) or "ARCHE_CHEM_API_KEY"
        )
        general_api_key_env = (
            getattr(args, "expert_general_api_key_env", None)
            or "ARCHE_CHEM_API_KEY"
        )
        fallback_api_key_env = (
            getattr(args, "expert_fallback_api_key_env", None)
            or "ARCHE_CHEM_API_KEY"
        )
        general_api_key = getattr(args, "api_key", None) or os.environ.get(
            "ARCHE_CHEM_API_KEY"
        )
        fallback_api_key = getattr(args, "api_key", None) or os.environ.get(
            "ARCHE_CHEM_API_KEY"
        )
    else:
        expert_api_key_env = "ARCHE_CHEM_API_KEY"
        general_api_key_env = "ARCHE_CHEM_API_KEY"
        fallback_api_key_env = "ARCHE_CHEM_API_KEY"
        general_api_key = None
        fallback_api_key = None

    runtime = build_expert_provider_bundle(
        enabled=expert_model_enabled,
        expert_model=getattr(args, "expert_model", None),
        expert_base_url=getattr(args, "expert_base_url", None),
        expert_api_key_env=expert_api_key_env,
        fallback_model=getattr(args, "expert_fallback_model", None),
        fallback_base_url=getattr(args, "expert_fallback_base_url", None),
        fallback_api_key=fallback_api_key,
        fallback_api_key_env=fallback_api_key_env,
        general_model=getattr(args, "expert_general_model", None),
        general_base_url=getattr(args, "expert_general_base_url", None),
        general_api_key=general_api_key,
        general_api_key_env=general_api_key_env,
    )

    return ResearchTaskApplication(
        deepseek_api_key=getattr(args, "api_key", None),
        work_dir=getattr(args, "work_dir", None),
        memory_enabled=not bool(getattr(args, "disable_memory", False)),
        memory_db_path=getattr(args, "memory_db", None),
        retrieval_embedder_name=_select_retrieval_embedder_name(
            not bool(getattr(args, "no_search", False))
        ),
        hypothesis_comparison_workers=getattr(
            args, "hypothesis_comparison_workers", "auto"
        ),
        hypothesis_generation_workers=getattr(
            args, "hypothesis_generation_workers", 1
        ),
        expert_model_runtime=runtime,
        case_profile=getattr(args, "case_profile", None),
        use_case1_author_main_hypothesis=bool(
            getattr(args, "use_case1_author_main_hypothesis", False)
        ),
        use_case3_author_main_hypothesis=bool(
            getattr(args, "use_case3_author_main_hypothesis", False)
        ),
        use_case3_author_main_hypothesis_advisory=bool(
            getattr(args, "use_case3_author_main_hypothesis_advisory", False)
        ),
        use_case3_author_main_hypothesis_exploration=bool(
            getattr(args, "use_case3_author_main_hypothesis_exploration", False)
        ),
    )



def run_from_args(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """解析六个公开动作、执行一次动作并返回结构化结果。"""

    parser = _build_cli_parser()
    args = parser.parse_args(argv)
    _validate_cli_request(parser, args)
    controller = _controller_from_cli_args(args)
    action = args.action

    if action == "resume-research-task":
        result = controller.resume_research_task(args.research_task_id)
    elif action == "get-hypothesis-confirmation":
        result = controller.get_hypothesis_confirmation(args.research_task_id)
    elif action == "submit-hypothesis-confirmation-and-continue":
        try:
            request, file_specs = _load_hypothesis_confirmation_request(
                args.confirmation_request_json
            )
            with ExitStack() as stack:
                files: dict[str, ResearchMaterialUpload] = {}
                for upload_ref, file_spec in file_specs.items():
                    descriptor = -1
                    try:
                        descriptor = os.open(
                            file_spec["path"],
                            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                        )
                        metadata = os.fstat(descriptor)
                        if not stat.S_ISREG(metadata.st_mode):
                            raise ValueError("confirmation request 只接受普通文件")
                        stream = stack.enter_context(os.fdopen(descriptor, "rb"))
                        descriptor = -1
                    except OSError as exc:
                        raise ValueError(
                            f"confirmation request files[{upload_ref}] 无法安全打开"
                        ) from exc
                    finally:
                        if descriptor >= 0:
                            os.close(descriptor)
                    files[upload_ref] = ResearchMaterialUpload(
                        filename=file_spec["filename"],
                        media_type=file_spec["media_type"],
                        content=stream,
                    )
                result = controller.submit_hypothesis_confirmation_and_continue(
                    args.research_task_id,
                    confirmation_id=request["confirmation_id"],
                    expected_research_task_sequence=request[
                        "expected_research_task_sequence"
                    ],
                    decisions=request["decisions"],
                    files=files,
                )
        except ValueError as exc:
            parser.error(str(exc))
    elif action == "create-author-hypothesis-task":
        try:
            author_hypotheses = _load_strict_author_hypotheses_json(
                args.author_hypotheses_json
            )
        except ValueError as exc:
            parser.error(str(exc))
        result = controller.create_author_hypothesis_task(
            scientific_question=args.question,
            author_hypotheses=author_hypotheses,
        )
    else:
        paper_search = normalize_paper_search_config(
            {
                "enabled": not args.no_search,
                "provider_strategy": "pypaperbot_scholar",
                "target_papers": args.paper_search_target_papers,
                "manifest_path": args.paper_search_manifest,
                "resume_state_path": args.paper_search_resume_state,
                "response_cache_dir": args.paper_search_cache_dir,
                "timeout_seconds": args.paper_search_timeout_seconds,
                "keyword_delay_seconds": args.paper_search_keyword_delay_seconds,
                "keyword_jitter_seconds": args.paper_search_keyword_jitter_seconds,
                "download_delay_seconds": args.paper_search_download_delay_seconds,
                "scholar_max_pages": args.paper_search_scholar_max_pages,
                "scholar_results_per_page": args.paper_search_scholar_results_per_page,
                "scholar_crossref_max_candidates": args.paper_search_scholar_crossref_max_candidates,
                "scholar_request_delay_seconds": args.paper_search_scholar_request_delay_seconds,
                "scholar_request_jitter_seconds": args.paper_search_scholar_request_jitter_seconds,
                "scholar_initial_delay_seconds": args.paper_search_scholar_initial_delay_seconds,
                "scholar_initial_jitter_seconds": args.paper_search_scholar_initial_jitter_seconds,
                "scholar_backoff_base_seconds": args.paper_search_scholar_backoff_base_seconds,
                "scholar_max_backoff_seconds": args.paper_search_scholar_max_backoff_seconds,
                "scholar_max_consecutive_rate_limits": args.paper_search_scholar_max_consecutive_rate_limits,
            },
            enabled=not args.no_search,
        )
        supplementary: Mapping[str, Any] | None = None
        scientific_input_path = getattr(args, "scientific_input_json", None)
        if scientific_input_path:
            try:
                supplementary = _load_strict_scientific_input_json(
                    scientific_input_path
                )
            except ValueError as exc:
                parser.error(str(exc))
        if action == "create-top5-confirmation-task":
            result = controller.create_top5_confirmation_task(
                scientific_question=args.question,
                num_queries=args.num_queries,
                num_hypotheses_per_query=args.num_hypotheses_per_query,
                top_n_hypotheses=args.top_n_hypotheses,
                pdf_dir=args.pdf_dir,
                index_dir=args.index_dir,
                search_papers=not args.no_search,
                paper_search=paper_search,
                supplementary_scientific_input=supplementary,
            )
        else:
            run_kwargs: Dict[str, Any] = {
                "scientific_question": args.question,
                "num_queries": args.num_queries,
                "num_hypotheses_per_query": args.num_hypotheses_per_query,
                "top_n_hypotheses": args.top_n_hypotheses,
                "pdf_dir": args.pdf_dir,
                "index_dir": args.index_dir,
                "search_papers": not args.no_search,
                "paper_search": paper_search,
                "max_parallel_investigations": args.max_parallel_investigations,
                "max_investigation_repairs_per_combination": (
                    args.max_investigation_repairs_per_combination
                ),
                "max_hypothesis_batches": args.max_hypothesis_batches,
                "supplementary_scientific_input": supplementary,
            }
            for flag_name in (
                "use_case1_author_main_hypothesis",
                "use_case3_author_main_hypothesis",
                "use_case3_author_main_hypothesis_advisory",
                "use_case3_author_main_hypothesis_exploration",
            ):
                if getattr(args, flag_name, False):
                    run_kwargs[flag_name] = True
            if getattr(args, "case3_prepared_input_dir", None):
                run_kwargs["case3_prepared_input_dir"] = args.case3_prepared_input_dir
            result = controller.create_standard_research_task(**run_kwargs)

    print(json.dumps(_cli_result_view(result), ensure_ascii=False))
    return result

def main(argv: Optional[List[str]] = None) -> int:
    """执行命令行任务，并把正式失败状态映射为非零退出码。"""

    result = run_from_args(argv)
    return 1 if result.get("status") in {"failed", "retrieval_incomplete"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
