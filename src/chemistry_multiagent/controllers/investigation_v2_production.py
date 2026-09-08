"""investigation_v2 公开入口的生产装配与结果投影。

本模块只在 Working Memory 的 Task/Module 边界上编排公开阶段。Investigation
and Experiment 的 thread、turn、Goal、scope、job、submission、receipt 和
sidecar 都属于 Harness adapter 的私有实现，不能成为本模块的路由依据。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from chemistry_multiagent.investigation.harness_runtime import (
    InvestigationHarnessRequest,
    SchemaV1LegacyInvestigationHarnessRequestReader,
)
from chemistry_multiagent.investigation.v2_run_bundle_projector import (
    InvestigationV2RunBundleProjector,
)
from chemistry_multiagent.investigation.research_material_store import (
    MaterialSizePolicy,
    ResearchMaterialStore,
)
from chemistry_multiagent.memory import ResearchTaskWorkingMemory
from chemistry_multiagent.memory.v2_migration import (
    ExistingHarnessRuntime,
    MigrationRecoveryEvidenceMissing,
)
from chemistry_multiagent.utils.run_bundle import (
    RunBundle,
    RESEARCH_TASK_STAGE_LAYOUT,
)

from .investigation_v2_module_controller import (
    ControllerAwaitingResearcherConfirmation,
    ControllerConfirmationReplayed,
    ControllerFailed,
    ControllerLeaseLostResult,
    ControllerSuspended,
    ResearchWorkflowController,
    ResearchTaskClosedResult,
    ResearchTaskFailedResult,
)
from .investigation_v2_confirmation import (
    HypothesisConfirmationService,
    ResearchMaterialUpload,
)


class ResearchTaskProductionError(RuntimeError):
    """公开 V2 生产装配无法安全创建或恢复任务。"""


V2_TASK_CONTRACT_VERSION = "investigation_v2.task.v2"
DEFAULT_RESEARCH_MATERIAL_POLICY: dict[str, int] = {
    "max_file_size_bytes": 64 * 1024 * 1024,
    "max_material_count": 25,
    "max_task_size_bytes": 256 * 1024 * 1024,
    "max_filename_bytes": 255,
}


def _normalize_research_material_policy(
    value: Any,
) -> dict[str, int]:
    """校验并复制冻结到 TaskCreated 的材料数量和大小策略。"""

    if value is None:
        return dict(DEFAULT_RESEARCH_MATERIAL_POLICY)
    if not isinstance(value, Mapping) or set(value) != set(
        DEFAULT_RESEARCH_MATERIAL_POLICY
    ):
        raise ResearchTaskProductionError(
            "research_material_policy 字段合同不匹配"
        )
    normalized: dict[str, int] = {}
    for field_name in DEFAULT_RESEARCH_MATERIAL_POLICY:
        raw = value[field_name]
        minimum = 1
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < minimum:
            raise ResearchTaskProductionError(
                f"research_material_policy.{field_name} 必须是正整数"
            )
        normalized[field_name] = raw
    if normalized["max_task_size_bytes"] < normalized["max_file_size_bytes"]:
        raise ResearchTaskProductionError(
            "research_material_policy 的任务总大小不能小于单文件大小"
        )
    return normalized


class PersistentHarnessRuntimeResolver:
    """只读核验迁移所需的既有 Harness runtime 身份。

    旧 ``investigation_v2`` source 只保存 Working Memory 事件，Open Module Run
    还需要从同一 task 的 Harness manifest 找回不可变 runtime 身份。resolver 只
    扫描规范 ``internal/state/harness/runs`` 下的 manifest，并按 task、模块类型和
    输入摘要唯一匹配；不创建目录、不调用 Harness、模型、MCP 或化学作业。零个或
    多个匹配都 fail closed，避免从 thread、scope、job 或文件名猜测 runtime。
    """

    _MANIFEST_KEYS = {
        "schema_version",
        "harness_run_ref",
        "request",
        "state",
        "failure_code",
    }
    _RUN_STATES = {"open", "suspended", "completed", "failed"}

    def __init__(self, runtime_root: str | Path) -> None:
        """绑定现有 Harness 目录；初始化阶段不创建或修改任何文件。"""

        if not isinstance(runtime_root, (str, Path)):
            raise TypeError("runtime_root 必须是路径")
        self.runtime_root = Path(runtime_root).expanduser().resolve(strict=False)

    def resolve(
        self,
        research_task_id: str,
        *,
        source_event_ids: tuple[str, ...] | list[str],
        expected_module_kind: str,
        expected_module_input_digest: str,
    ) -> ExistingHarnessRuntime:
        """从 manifest 唯一恢复 runtime；任何证据缺失或歧义均抛出迁移错误。"""

        if not isinstance(research_task_id, str) or not research_task_id.strip():
            raise MigrationRecoveryEvidenceMissing(
                "runtime resolver 缺少 research_task_id"
            )
        if not source_event_ids:
            raise MigrationRecoveryEvidenceMissing(
                "runtime resolver 缺少 source event 身份"
            )
        if expected_module_kind != "hypothesis-investigation":
            raise MigrationRecoveryEvidenceMissing(
                "runtime resolver 只支持 hypothesis-investigation module"
            )
        if (
            not isinstance(expected_module_input_digest, str)
            or not expected_module_input_digest
        ):
            raise MigrationRecoveryEvidenceMissing(
                "runtime resolver 缺少 module_input_digest"
            )
        runs_root = self.runtime_root / "runs"
        if not runs_root.is_dir() or runs_root.is_symlink():
            raise MigrationRecoveryEvidenceMissing(
                "既有 Harness runtime 目录不存在"
            )
        candidates: list[ExistingHarnessRuntime] = []
        for manifest_path in sorted(runs_root.rglob("manifest.json")):
            candidate = self._read_candidate(
                manifest_path,
                research_task_id=research_task_id,
                expected_module_input_digest=expected_module_input_digest,
            )
            if candidate is not None:
                candidates.append(candidate)
        if not candidates:
            raise MigrationRecoveryEvidenceMissing(
                "没有唯一 manifest 能证明既有 Investigation runtime"
            )
        if len(candidates) != 1:
            raise MigrationRecoveryEvidenceMissing(
                "多个 manifest 匹配同一 Open Investigation runtime"
            )
        return candidates[0]

    def _read_candidate(
        self,
        manifest_path: Path,
        *,
        research_task_id: str,
        expected_module_input_digest: str,
    ) -> ExistingHarnessRuntime | None:
        """读取一个候选 manifest，非匹配项返回 None，匹配损坏则停止。"""

        if manifest_path.is_symlink() or any(
            parent.is_symlink() for parent in manifest_path.parents
            if parent != self.runtime_root
        ):
            raise MigrationRecoveryEvidenceMissing(
                "Harness manifest 路径经过符号链接"
            )
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, TypeError, ValueError) as exc:
            raise MigrationRecoveryEvidenceMissing(
                "Harness manifest 无法读取"
            ) from exc
        if not isinstance(raw, Mapping):
            raise MigrationRecoveryEvidenceMissing("Harness manifest 必须是 object")
        if set(raw) != self._MANIFEST_KEYS or raw.get("schema_version") != 1:
            raise MigrationRecoveryEvidenceMissing("Harness manifest 字段合同损坏")
        harness_run_ref = raw.get("harness_run_ref")
        if not isinstance(harness_run_ref, str) or not harness_run_ref:
            raise MigrationRecoveryEvidenceMissing(
                "Harness manifest 缺少 harness_run_ref"
            )
        invocation_dir = manifest_path.parent
        run_dir = (
            invocation_dir.parent
            if invocation_dir.parent.name == "invocations"
            else invocation_dir
        )
        expected_directory = hashlib.sha256(
            harness_run_ref.encode("utf-8")
        ).hexdigest()
        if run_dir.parent != self.runtime_root / "runs":
            raise MigrationRecoveryEvidenceMissing(
                "Harness manifest 不在规范 runtime 目录"
            )
        if run_dir.name != expected_directory:
            raise MigrationRecoveryEvidenceMissing(
                "Harness manifest 目录与 harness_run_ref 不一致"
            )
        state = raw.get("state")
        if state not in self._RUN_STATES:
            raise MigrationRecoveryEvidenceMissing("Harness manifest 状态损坏")
        request_mapping = raw.get("request")
        try:
            request = InvestigationHarnessRequest.from_mapping(request_mapping)
        except (KeyError, TypeError, ValueError):
            try:
                request = SchemaV1LegacyInvestigationHarnessRequestReader.read(
                    request_mapping
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise MigrationRecoveryEvidenceMissing(
                    "Harness manifest request 已损坏"
                ) from exc
        if request.research_task_id != research_task_id:
            return None
        if request.module_kind != "hypothesis-investigation":
            return None
        if request.module_input_digest != expected_module_input_digest:
            return None
        context_binding = request.context_binding
        owner_id = (
            context_binding.get("investigation_context_id")
            if isinstance(context_binding, Mapping)
            else None
        )
        if not isinstance(owner_id, str) or not owner_id.strip():
            owner_id = (
                context_binding.get("identity")
                if isinstance(context_binding, Mapping)
                else None
            )
        if not isinstance(owner_id, str) or not owner_id.strip():
            raise MigrationRecoveryEvidenceMissing(
                "Harness manifest 缺少可转换为 context_binding 的 owner"
            )
        return ExistingHarnessRuntime(
            runtime_ref=owner_id,
            harness_run_ref=harness_run_ref,
            state=state,
        )


def _safe_research_task_id(research_task_id: Any) -> str:
    """校验 research_task 身份可用于规范 Bundle 路径。"""

    if not isinstance(research_task_id, str) or not research_task_id.strip():
        raise ResearchTaskProductionError("research_task_id 必须是非空字符串")
    normalized = research_task_id.strip()
    if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ResearchTaskProductionError(
            "research_task_id 不能包含路径分隔符"
        )
    return normalized


class ResearchTaskProductionAssembly:
    """把公开 Controller 输入装配到 V2 ledger/controller。

    ``module_adapters`` 是窄的公开模块替代实现接缝；未提供时由调用方传入的
    ``default_module_factory`` 装配真实 Retrieval、Hypothesis、Harness 和 Review
    adapter。此类本身不解释任何 Harness 私有状态。
    """

    def __init__(
        self,
        controller: Any,
        *,
        module_adapters: Mapping[str, Any] | None = None,
        default_module_factory: Any | None = None,
    ) -> None:
        self.controller = controller
        self.module_adapters = (
            None if module_adapters is None else dict(module_adapters)
        )
        self.default_module_factory = default_module_factory

    @property
    def storage_root(self) -> Path:
        """返回新任务和恢复任务共用的持久根目录。"""

        return Path(
            getattr(
                self.controller,
                "_run_storage_root",
                getattr(self.controller, "work_dir", "."),
            )
        ).expanduser().resolve(strict=False)

    def _database_path_for_task(self, research_task_id: str) -> Path:
        """按显式数据库或规范 Bundle 路径定位 V2 ledger。"""

        explicit = getattr(self.controller, "memory_db_path", None)
        if getattr(self.controller, "_memory_db_explicit", False) and explicit:
            return Path(explicit).expanduser().resolve(strict=False)
        return (
            self.storage_root
            / "outputs"
            / "research_tasks"
            / _safe_research_task_id(research_task_id)
            / "internal"
            / "state"
            / "arche_memory.sqlite3"
        )

    def _new_bundle(
        self,
        research_task_id: str,
        *,
        scientific_question: str,
        task_config: Mapping[str, Any],
    ) -> RunBundle:
        """创建使用 V2 阶段布局的 Run Bundle。"""

        return RunBundle.create_research_task(
            self.storage_root,
            _safe_research_task_id(research_task_id),
            scientific_question=scientific_question,
            task_config=dict(task_config),
            model_configuration={"architecture_version": "investigation_v2"},
        )

    def _controller(
        self,
        service: ResearchTaskWorkingMemory,
        *,
        task_root: Path,
        task_input: Mapping[str, Any],
        policy: Mapping[str, Any],
        dependencies: Mapping[str, Any],
        task_config: Mapping[str, Any],
    ) -> ResearchWorkflowController:
        """构造唯一 V2 模块 Controller。"""

        if policy.get("stage_architecture_version") != "investigation_v2":
            raise ResearchTaskProductionError(
                "V2 TaskCreated 架构版本缺失或不是 investigation_v2"
            )
        material_store = self._research_material_store(task_root, policy)
        modules = self.module_adapters
        if modules is None:
            if not callable(self.default_module_factory):
                raise ResearchTaskProductionError(
                    "V2 生产入口缺少默认模块装配 factory"
                )
            modules = self.default_module_factory(
                task_root=task_root,
                working_memory=service,
                task_input=task_input,
                policy=policy,
                dependencies=dependencies,
                task_config=task_config,
                research_material_store=material_store,
            )
        if not isinstance(modules, Mapping):
            raise ResearchTaskProductionError("V2 模块装配必须返回 mapping")
        return ResearchWorkflowController(
            service,
            modules,
            require_entry_strategy=True,
            material_store=material_store,
        )

    @staticmethod
    def _research_material_store(
        task_root: Path,
        policy: Mapping[str, Any],
    ) -> ResearchMaterialStore | None:
        """按 TaskCreated 冻结策略重建同一个任务级材料字节 owner。

        新 ``task.v2`` 任务必须带完整策略；没有该字段的历史 ``task.v1`` 保持原
        恢复行为，不会被当前默认值静默升级。构造 store 只核对 task 根，不创建
        CAS；只有研究员实际上传材料时才会写入字节。
        """

        raw_policy = policy.get("research_material_policy")
        if raw_policy is None:
            return None
        normalized = _normalize_research_material_policy(raw_policy)
        return ResearchMaterialStore(
            task_root,
            MaterialSizePolicy(**normalized),
        )

    @staticmethod
    def _result(
        result: Any,
        facts: Any,
    ) -> dict[str, Any]:
        """把 V2 Controller 结果投影为公开入口摘要。"""

        research_task_id = facts.research_task_id
        payload: dict[str, Any] = {
            "research_task_id": research_task_id,
            "architecture_version": "investigation_v2",
            "memory_status": "available",
            "status": "open",
        }
        if isinstance(result, (ResearchTaskClosedResult, ResearchTaskFailedResult)):
            payload["status"] = result.outcome.get("status")
            payload["outcome"] = dict(result.outcome)
        elif isinstance(result, ControllerAwaitingResearcherConfirmation):
            confirmation = result.to_public_dict()
            payload.update(confirmation)
            payload["research_task_id"] = research_task_id
            payload["architecture_version"] = "investigation_v2"
            payload["memory_status"] = "available"
        elif isinstance(result, ControllerConfirmationReplayed):
            payload.update(
                {
                    "status": "open",
                    "reason_code": result.reason_code,
                    "confirmation_id": result.confirmation_id,
                }
            )
        elif isinstance(result, ControllerSuspended):
            payload.update(
                {
                    "status": "open",
                    "reason_code": result.reason_code,
                    "module_invocation_id": result.module_invocation_id,
                    "recovery_status": result.recovery_verification.status.value,
                }
            )
        elif isinstance(result, ControllerLeaseLostResult):
            payload.update(
                {
                    "status": "open",
                    "reason_code": result.reason_code,
                    "module_invocation_id": result.module_invocation_id,
                }
            )
        elif isinstance(result, ControllerFailed):
            payload.update(
                {
                    "status": "failed",
                    "reason_code": result.failure_code,
                    "module_invocation_id": result.module_invocation_id,
                }
            )
        else:
            raise ResearchTaskProductionError(
                f"V2 Controller 返回了未注册结果：{type(result).__name__}"
            )
        return payload

    def create_research_task(
        self,
        *,
        research_task_id: str,
        scientific_question: str,
        task_config: Mapping[str, Any],
        task_input: Mapping[str, Any],
        policy: Mapping[str, Any],
        dependencies: Mapping[str, Any],
        contract_version: str = V2_TASK_CONTRACT_VERSION,
    ) -> dict[str, Any]:
        """创建 V2 Task、运行公开模块链并更新展示投影。

        新建任务默认使用 ``investigation_v2.task.v2``，该版本表示 Selection 或
        Hypothesis Import 后进入统一研究员确认边界。历史 ``task.v1`` 只在既有
        ledger 的恢复路径中读取，不能由这个新任务入口重新创建。
        """

        research_task_id = _safe_research_task_id(research_task_id)
        frozen_policy = dict(policy)
        if contract_version == V2_TASK_CONTRACT_VERSION:
            frozen_policy["research_material_policy"] = (
                _normalize_research_material_policy(
                    frozen_policy.get("research_material_policy")
                )
            )
        bundle = self._new_bundle(
            research_task_id,
            scientific_question=scientific_question,
            task_config=task_config,
        )
        service = ResearchTaskWorkingMemory(
            str(self._database_path_for_task(research_task_id)),
            enabled=True,
            scratch_root=str(self.storage_root),
        )
        module_controller = self._controller(
            service,
            task_root=bundle.root,
            task_input=task_input,
            policy=frozen_policy,
            dependencies=dependencies,
            task_config=task_config,
        )
        result = module_controller.run_research_task(
            research_task_id,
            contract_version=contract_version,
            input=task_input,
            policy=frozen_policy,
            dependencies=dependencies,
        )
        facts = service.read_research_task_facts(research_task_id)
        InvestigationV2RunBundleProjector(bundle, service).project(
            facts,
            controller_present=True,
        )
        return self._result(result, facts)

    def resume_research_task(self, research_task_id: str) -> dict[str, Any]:
        """从规范 V2 ledger 恢复同一个 research_task。"""

        research_task_id = _safe_research_task_id(research_task_id)
        db_path = self._database_path_for_task(research_task_id)
        if not db_path.is_file():
            raise ResearchTaskProductionError("V2 Task ledger 不存在")
        bundle = RunBundle.open_existing_research_task(
            self.storage_root,
            research_task_id,
        )
        service = ResearchTaskWorkingMemory(
            str(db_path), enabled=True, scratch_root=str(self.storage_root)
        )
        facts_before = service.read_research_task_facts(research_task_id)
        frozen_task_config = facts_before.research_task.input.get("task_config")
        if not isinstance(frozen_task_config, Mapping):
            frozen_task_config = facts_before.research_task.policy.get(
                "task_config", {}
            )
        module_controller = self._controller(
            service,
            task_root=bundle.root,
            task_input=facts_before.research_task.input,
            policy=facts_before.research_task.policy,
            dependencies=facts_before.research_task.dependencies,
            task_config=frozen_task_config,
        )
        result = module_controller.resume_research_task(research_task_id)
        facts = service.read_research_task_facts(research_task_id)
        InvestigationV2RunBundleProjector(bundle, service).project(
            facts,
            controller_present=True,
        )
        return self._result(result, facts)

    def get_hypothesis_confirmation(
        self,
        research_task_id: str,
    ) -> dict[str, Any]:
        """只读返回同一任务当前的研究员确认视图。

        该入口只读取 V2 ledger，不装配模型、Catalog、scope 或 Investigation
        Session，也不改写 Run Bundle。
        """

        research_task_id = _safe_research_task_id(research_task_id)
        db_path = self._database_path_for_task(research_task_id)
        if not db_path.is_file():
            raise ResearchTaskProductionError("V2 Task ledger 不存在")
        service = ResearchTaskWorkingMemory(
            str(db_path), enabled=True, scratch_root=str(self.storage_root)
        )
        return HypothesisConfirmationService(service).read_view(research_task_id).to_dict()

    def submit_hypothesis_confirmation_and_continue(
        self,
        research_task_id: str,
        *,
        confirmation_id: str,
        expected_research_task_sequence: int,
        decisions: Sequence[Mapping[str, Any]],
        files: Mapping[str, ResearchMaterialUpload] | None = None,
    ) -> dict[str, Any]:
        """原子提交研究员决定，并从同一 Task Facts 继续执行已选假设。

        ``files`` 只携带上传引用和文件流；调用方本机路径不会写入 Working
        Memory。确认成功后才装配后续模块并允许创建 Session、scope 和模型调用。
        """

        research_task_id = _safe_research_task_id(research_task_id)
        db_path = self._database_path_for_task(research_task_id)
        if not db_path.is_file():
            raise ResearchTaskProductionError("V2 Task ledger 不存在")
        bundle = RunBundle.open_existing_research_task(
            self.storage_root,
            research_task_id,
        )
        service = ResearchTaskWorkingMemory(
            str(db_path), enabled=True, scratch_root=str(self.storage_root)
        )
        facts_before = service.read_research_task_facts(research_task_id)
        frozen_task_config = facts_before.research_task.input.get("task_config")
        if not isinstance(frozen_task_config, Mapping):
            frozen_task_config = facts_before.research_task.policy.get(
                "task_config", {}
            )
        module_controller = self._controller(
            service,
            task_root=bundle.root,
            task_input=facts_before.research_task.input,
            policy=facts_before.research_task.policy,
            dependencies=facts_before.research_task.dependencies,
            task_config=frozen_task_config,
        )
        result = module_controller.submit_hypothesis_confirmation_and_continue(
            research_task_id,
            confirmation_id=confirmation_id,
            expected_research_task_sequence=expected_research_task_sequence,
            decisions=decisions,
            files=files,
        )
        facts = service.read_research_task_facts(research_task_id)
        InvestigationV2RunBundleProjector(bundle, service).project(
            facts,
            controller_present=True,
        )
        return self._result(result, facts)

__all__ = [
    "DEFAULT_RESEARCH_MATERIAL_POLICY",
    "ResearchTaskProductionAssembly",
    "ResearchTaskProductionError",
    "PersistentHarnessRuntimeResolver",
    "V2_TASK_CONTRACT_VERSION",
]
