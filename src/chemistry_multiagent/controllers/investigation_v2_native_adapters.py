"""V2 Investigation/Review 的默认 native 装配。

这个模块只负责把公开 V2 记录接到两个深模块：

* ``InvestigationContextRunner`` 负责五个假设 Session 的编排；
* ``ResearchTaskReviewRunner`` 负责只读 Result heads 的 DS 审查。

Session、thread、scope、job 和 receipt 不会写入 V2 Working Memory。生产 transport
可以由 ``dependencies`` 注入；未注入时这里按冻结任务输入懒加载 Codex/MiniChem，因而
构造 adapter 本身不会启动进程、连接 MCP 或提交作业。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import os
import re
import secrets
import threading
from pathlib import Path, PurePosixPath
from typing import Any

from chemistry_multiagent.harness.codex_process_sandbox import CodexProcessSandboxSpec
from chemistry_multiagent.harness.session_store import CodexSessionOwner, CodexSessionStore
from chemistry_multiagent.investigation.context_backend import (
    InvestigationContextRunner,
    InvestigationSessionTemporarilyUnavailable,
)
from chemistry_multiagent.investigation.context_store import (
    InvestigationContextIdentity,
    InvestigationContextStore,
    InvestigationSessionIdentity,
)
from chemistry_multiagent.investigation.codex_runtime import (
    EvidenceMcpSettings,
    InvestigationCodexRuntime,
    InvestigationMcpSettings,
)
from chemistry_multiagent.investigation.harness_recovery_store import (
    HarnessRunIdentity,
    SessionReference,
)
from chemistry_multiagent.investigation.harness_runtime import (
    InvestigationHarness,
    InvestigationHarnessRequest,
)
from chemistry_multiagent.investigation.investigation_backend import (
    EvidenceResultRejected,
    NativeAction,
    NativeExecutionTransport,
    NativeProviderTransport,
    InvestigationSessionRunner,
)
from chemistry_multiagent.investigation.review_runtime import (
    ResearchTaskReviewRunner,
    V2TaskReviewModuleAdapter,
    build_v2_evidence_snapshot_builder,
)
from chemistry_multiagent.investigation.research_material_store import (
    ResearchMaterialStore,
)
from chemistry_multiagent.investigation.evidence import ReviewEvidenceStore
from chemistry_multiagent.investigation.evidence_mcp_lease import TaskEvidenceMcpLease
from chemistry_multiagent.investigation.evidence_mcp_server import (
    DEFAULT_EVIDENCE_CAPABILITY_HEADER,
)
from chemistry_multiagent.investigation.domain import InvestigationResult, InvestigationSession
from chemistry_multiagent.investigation.minichem_gateway import (
    InvestigationMiniChemGateway,
    InvestigationMiniChemReferenceRejected,
    _AuthenticatedStreamableHTTPContext,
)
from chemistry_multiagent.investigation.scope_credentials import (
    InvestigationScopeCredentialStore,
)
from chemistry_multiagent.tools.minichem_catalog_freeze import (
    FrozenMiniChemCatalog,
    MiniChemCatalogFreezeError,
    freeze_minichem_catalog,
    load_frozen_minichem_catalog_startup,
)
from chemistry_multiagent.memory import (
    HYPOTHESIS_CONFIRMATION_V2_SCHEMA_VERSION,
    ResearchTaskWorkingMemory,
    ScientificRecord,
)
from chemistry_multiagent.tools.minichem_adapter import (
    MiniChemMcpAdapter,
    MiniChemMcpError,
)

from .investigation_v2_module_controller import HarnessModuleRunner


def _digest(value: Any) -> str:
    """计算稳定摘要，用于派生不透明运行身份。"""

    import json

    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _investigation_endpoint(dependencies: Mapping[str, Any]) -> str:
    """读取 Investigation 专用 9012 endpoint；旧环境变量仅作边界兼容。"""

    return str(
        dependencies.get(
            "minichem_endpoint",
            os.environ.get(
                "MINICHEM_INVESTIGATION_ENDPOINT",
                "http://127.0.0.1:9012/mcp",
            ),
        )
    )


def _safe_token(value: Any, label: str) -> str:
    """校验 transport 返回的公开身份，不生成空身份。"""

    if not isinstance(value, str) or not value.strip() or "/" in value or "\\" in value:
        raise ValueError(f"{label} 无效")
    return value.strip()


def _resolve_investigation_codex_thread_dependency(
    dependencies: Mapping[str, Any],
) -> tuple[str | None, int | None]:
    """读取新任务冻结的 Investigation Codex thread 配置。

    输入是 ``TaskCreated.dependencies``；动作核对压缩提示词摘要和 1M 档位的
    正整数配置，不读取 provider 或环境。返回可交给每个 Investigation runtime
    的提示词与窗口；历史任务没有该结构时返回两个 ``None``，保持原 thread
    策略。字段缺失、额外字段或摘要漂移时立即失败。
    """

    raw_policy = dependencies.get("investigation_codex_thread")
    if raw_policy is None:
        return None, None
    if not isinstance(raw_policy, Mapping):
        raise TypeError("investigation_codex_thread 必须是 object")
    expected_fields = {
        "compact_prompt",
        "compact_prompt_sha256",
        "model_context_window",
    }
    if set(raw_policy) != expected_fields:
        raise ValueError("investigation_codex_thread 字段不完整或包含未知字段")
    compact_prompt = raw_policy.get("compact_prompt")
    if not isinstance(compact_prompt, str) or not compact_prompt.strip():
        raise ValueError("investigation_codex_thread.compact_prompt 必须是非空文本")
    compact_prompt = compact_prompt.strip()
    compact_prompt_sha256 = raw_policy.get("compact_prompt_sha256")
    expected_sha256 = hashlib.sha256(compact_prompt.encode("utf-8")).hexdigest()
    if compact_prompt_sha256 != expected_sha256:
        raise ValueError("investigation_codex_thread 压缩提示词摘要漂移")
    model_context_window = raw_policy.get("model_context_window")
    if (
        isinstance(model_context_window, bool)
        or not isinstance(model_context_window, int)
        or model_context_window <= 0
    ):
        raise ValueError("investigation_codex_thread.model_context_window 必须是正整数")
    return compact_prompt, model_context_window


def _canonical_session_workspace_relpath(
    *,
    investigation_session_id: str,
    research_task_id: str,
    investigation_context_id: str,
    hypothesis_execution_input_record_id: str,
) -> str:
    """返回当前 Run Bundle 内唯一的 Investigation workspace 相对路径。"""

    return (
        PurePosixPath("internal")
        / "state"
        / "investigation_runtime"
        / "sessions"
        / investigation_session_id
        / "codex_sessions"
        / "investigations"
        / research_task_id
        / investigation_context_id
        / hypothesis_execution_input_record_id
        / "workspace"
    ).as_posix()


def _session_owner_workspace(
    task_root: Path,
    session: InvestigationSessionIdentity,
) -> Path:
    """核对持久 workspace 身份并返回 scope、Codex 与 Session 共用目录。"""

    canonical = _canonical_session_workspace_relpath(
        investigation_session_id=session.investigation_session_id,
        research_task_id=session.research_task_id,
        investigation_context_id=session.investigation_context_id,
        hypothesis_execution_input_record_id=(
            session.hypothesis_execution_input_record_id
        ),
    )
    if session.workspace_identity != canonical:
        raise ValueError(
            "Investigation Session workspace 身份与当前 Run Bundle 目录不一致"
        )
    root = Path(task_root).resolve(strict=False)
    workspace = (root / canonical).resolve(strict=False)
    try:
        workspace.relative_to(root)
    except ValueError as exc:  # pragma: no cover - Context Store 已限制安全 ID
        raise ValueError("Investigation Session workspace 越过任务根目录") from exc
    return workspace


class _NativeScopeCapabilityManager:
    """为 native Investigation 持久化并复用 9012 scope capability。"""

    def __init__(
        self,
        *,
        task_root: Path,
        research_task_id_hint: str | None = None,
    ) -> None:
        registry_root = os.environ.get("MINICHEM_SCOPE_REGISTRY_ROOT")
        if not isinstance(registry_root, str) or not registry_root.strip():
            raise RuntimeError(
                "native Investigation 缺少 MINICHEM_SCOPE_REGISTRY_ROOT"
            )
        from minichem_toolbox.scope import ScopeCapabilityRegistry

        root = Path(task_root).resolve(strict=False)
        # outputs/research_tasks/<research_task_id> 对应 runtime 根为 parents[2]；
        # 测试目录没有该布局时退回 task root 的父目录，仍不把 token 写入 Bundle。
        runtime_root = (
            root.parents[2]
            if len(root.parents) >= 3 and root.parent.name == "research_tasks"
            else root.parent
        )
        research_task_id = research_task_id_hint or root.name
        credential_root = Path(
            os.environ.get(
                "MINICHEM_SCOPE_CREDENTIALS_ROOT",
                str(
                    runtime_root
                    / "minichem_scope_credentials"
                    / research_task_id
                ),
            )
        )
        self._store = InvestigationScopeCredentialStore(
            credential_root,
            registry=ScopeCapabilityRegistry(registry_root),
        )
        self._task_root = root
        self._caps: dict[str, Any] = {}
        self._lock = threading.RLock()

    def ensure_session(self, session: InvestigationSessionIdentity) -> None:
        """按完整身份创建或复用 capability，并校验私有 workspace。"""

        workspace = _session_owner_workspace(self._task_root, session)
        workspace.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(workspace, 0o700)
        with self._lock:
            capability = self._store.provision_investigation_scope_capability(
                task_id=session.research_task_id,
                batch_id=session.investigation_context_id,
                combination_id=session.hypothesis_execution_input_record_id,
                scope_id=session.minichem_scope_id,
                workspace=workspace,
            )
            self._caps[session.minichem_scope_id] = capability

    def capability_for(self, session: InvestigationSessionIdentity) -> Any:
        """返回已登记 scope 的明文 token，仅供当前 MCP session factory 使用。"""

        with self._lock:
            capability = self._caps.get(session.minichem_scope_id)
        if capability is None:
            self.ensure_session(session)
            with self._lock:
                capability = self._caps.get(session.minichem_scope_id)
        if capability is None:
            raise RuntimeError("native Investigation scope capability 不可用")
        return capability


class _FrozenCatalogSubmissionContractChecker:
    """按冻结 Catalog 的关键 Action/Backend 身份检查一次新提交。"""

    def __init__(self, frozen: FrozenMiniChemCatalog) -> None:
        self.catalog_hash = frozen.catalog_hash
        self.catalog_version = str(frozen.snapshot.get("schema_version", ""))
        contracts = frozen.snapshot.get("action_contracts", {})
        values = contracts.get("contracts", []) if isinstance(contracts, Mapping) else []
        self._pairs = {
            (item.get("action_id"), item.get("backend_id"))
            for item in values
            if isinstance(item, Mapping)
        }
        self._backends_by_action: dict[str, tuple[str, ...]] = {}
        for action in frozen.snapshot.get("actions", []):
            if isinstance(action, Mapping) and isinstance(action.get("id"), str):
                backends = action.get("backend_ids", [])
                if isinstance(backends, list):
                    self._backends_by_action[action["id"]] = tuple(
                        item for item in backends if isinstance(item, str)
                    )

    def check_submission_contract(
        self,
        action: NativeAction,
        *,
        session: SessionReference,
    ) -> bool:
        """只核对冻结 pair 和可选 catalog_hash，不执行全局服务探针。"""

        del session
        payload = action.payload
        declared_hash = payload.get("catalog_hash")
        if declared_hash is not None and declared_hash != self.catalog_hash:
            return False
        action_id = payload.get("action_id", action.action_id)
        if not isinstance(action_id, str) or not action_id.strip():
            return False
        backends = self._backends_by_action.get(action_id)
        if backends is None:
            return False
        backend_id = payload.get("backend_id", payload.get("backend"))
        if backend_id is None:
            # NativeJobRequest 用 software_id 表示实际执行后端；只有单一候选时
            # 才允许无歧义推断，避免模型偷偷切换 Backend。
            backend_id = payload.get("software_id")
        if backend_id is None and len(backends) == 1:
            backend_id = backends[0]
        return isinstance(backend_id, str) and (action_id, backend_id) in self._pairs


class _NativeEvidenceVerifier:
    """把 9012 权威引用核验转换为 V2 backend 的窄 evidence 摘要。"""

    def __init__(
        self,
        *,
        gateway: InvestigationMiniChemGateway,
        session: SessionReference,
        recovery_store: Any,
        artifact_manifest_registry: dict[str, Any],
    ) -> None:
        self.gateway = gateway
        self.session_ref = session
        self.recovery_store = recovery_store
        self.artifact_manifest_registry = artifact_manifest_registry

    def verify_evidence(
        self,
        response: Any,
        *,
        request: Any,
        jobs: Sequence[Any],
        prepared_output: Any,
    ) -> Mapping[str, Any]:
        """核对实际结果引用、分页读取 artifact 并登记 Review manifest。"""

        del prepared_output
        if response is None:
            raise RuntimeError("Investigation response 缺失")
        snapshot = self.recovery_store.read_snapshot(
            HarnessRunIdentity(
                research_task_id=request.research_task_id,
                module_invocation_id=request.module_invocation_id,
                harness_invocation_id=request.harness_invocation_id,
                module_input_digest=request.module_input_digest,
            )
        )
        attempt = snapshot.last_attempt
        if attempt is None:
            raise RuntimeError("evidence verifier 缺少 provider attempt")
        response_value = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        job_submission_operation_ids = {
            operation_id
            for job in jobs
            if isinstance(
                operation_id := (
                    job.get("operation_id")
                    if isinstance(job, Mapping)
                    else getattr(job, "operation_id", None)
                ),
                str,
            )
        }
        typed_operation_refs = tuple(
            operation_ref
            for operation_ref in response_value.get("operation_refs", ())
            if operation_ref not in job_submission_operation_ids
        )
        result = InvestigationResult(
            task_id=self.session_ref.research_task_id,
            batch_id=self.session_ref.investigation_context_id,
            combination_id=self.session_ref.hypothesis_execution_input_record_id,
            result_version="v1",
            codex_thread_id=attempt.model_thread_id,
            scope_id=self.session_ref.minichem_scope_id,
            turn_id=attempt.model_turn_id,
            self_judgment=response_value["self_judgment"],
            narrative=response_value["narrative"],
            job_refs=tuple(response_value.get("job_refs", ())),
            # 异步 job 的 operation_id 是提交来源，已由 job evidence 核对；
            # 其余引用才是需要作为独立 typed operation 严格验证的对象。
            operation_refs=typed_operation_refs,
            artifact_refs=tuple(response_value.get("artifact_refs", ())),
            observations=tuple(response_value.get("observations", ())),
            limitations=tuple(response_value.get("limitations", ())),
        )
        try:
            verified = self.gateway.verify_investigation_result_with_evidence(
                InvestigationSession(
                    task_id=self.session_ref.research_task_id,
                    batch_id=self.session_ref.investigation_context_id,
                    combination_id=self.session_ref.hypothesis_execution_input_record_id,
                    codex_thread_id=attempt.model_thread_id,
                    workspace_ref=self.session_ref.workspace_identity,
                    scope_id=self.session_ref.minichem_scope_id,
                ),
                result,
            )
        except InvestigationMiniChemReferenceRejected as error:
            raise EvidenceResultRejected(
                reason_code="evidence_references_rejected",
                invalid_references=error.invalid_references,
            ) from None
        manifests = self.gateway.materialize_result_artifacts(
            InvestigationSession(
                task_id=self.session_ref.research_task_id,
                batch_id=self.session_ref.investigation_context_id,
                combination_id=self.session_ref.hypothesis_execution_input_record_id,
                codex_thread_id=attempt.model_thread_id,
                workspace_ref=self.session_ref.workspace_identity,
                scope_id=self.session_ref.minichem_scope_id,
            ),
            result,
        )
        for source_ref, manifest in manifests.items():
            self.artifact_manifest_registry[source_ref] = manifest.to_public_dict()
        payload = verified.evidence.to_payload()
        digest = payload["verification_digest"]
        return {
            "verification_ref": payload["verification_object_id"],
            "verification_digest": digest,
            "source_manifest_ref": f"minichem-evidence-{digest[:24]}",
            "summary": result.narrative,
            "evidence_items": [
                {
                    "job_ref": item["job_id"],
                    "status": item["terminal_status"],
                }
                for item in payload.get("jobs", [])
            ],
            "verification": {
                "status": "verified",
                "source": "minichem-authority-adapter",
                "verification_ref": payload["verification_object_id"],
                "verification_digest": digest,
            },
            "artifact_refs": list(result.artifact_refs),
        }


class InvestigationSessionOwner:
    """按 Context 和最终执行 owner 确定性规划 Session。"""

    def __init__(self, *, task_root: Path, policy: Mapping[str, Any], dependencies: Mapping[str, Any]) -> None:
        self.task_root = Path(task_root)
        self.policy = copy.deepcopy(dict(policy))
        # dependency mapping 可能包含带锁的 provider/transport 对象；只复制 mapping
        # 容器，不能对这些外部 owner 做深拷贝。策略仍然冻结为独立 JSON 副本。
        self.dependencies = dict(dependencies)

    def plan_session(
        self,
        context: InvestigationContextIdentity,
        selected_hypothesis: ScientificRecord,
    ) -> InvestigationSessionIdentity:
        """只依据冻结科学记录派生 Session，不产生外部副作用。"""

        payload = selected_hypothesis.payload
        if not isinstance(payload, Mapping):
            raise ValueError("Session owner payload 非法")
        if selected_hypothesis.scientific_record_type == "HypothesisExecutionInputRecord":
            hypothesis = payload.get("hypothesis")
            if not isinstance(hypothesis, Mapping):
                raise ValueError("HypothesisExecutionInputRecord 缺少 hypothesis")
            hypothesis_ref = hypothesis.get("hypothesis_id", hypothesis.get("hypothesis_ref"))
            if not isinstance(hypothesis_ref, str) or not hypothesis_ref:
                hypothesis_ref = _digest(hypothesis)
            seed_material = {
                "research_task_id": context.research_task_id,
                "context": context.investigation_context_id,
                "execution_input": selected_hypothesis.scientific_record_id,
                "hypothesis": hypothesis_ref,
            }
        else:
            ref = payload.get("hypothesis_ref")
            if not isinstance(ref, str) or not ref:
                raise ValueError("SelectedHypothesisRecord 缺少 hypothesis_ref")
            seed_material = {
                "research_task_id": context.research_task_id,
                "context": context.investigation_context_id,
                "hypothesis": ref,
            }
        seed = _digest(seed_material)
        scope_contract = self.dependencies.get("scope_contract_digest")
        if scope_contract is None:
            scope_contract = self.dependencies.get(
                "investigation_scope_contract_digest"
            )
        if scope_contract is None:
            # 旧 dependency 名称仅作为输入边界兼容；绝不解释为 capability
            # generation digest。
            scope_contract = self.dependencies.get("minichem_scope_contract_digest")
        if scope_contract is None:
            # Context/Task policy 是合同摘要的最终 owner；它与后面由 capability
            # owner 返回的 generation digest 处于不同身份层，不能交叉回填。
            scope_contract = self.policy.get("scope_contract_digest")
        if not isinstance(scope_contract, str) or len(scope_contract) != 64:
            scope_contract = _digest(
                {"scope_contract": "investigation", "policy": self.policy}
            )
        investigation_session_id = f"investigation-session-{seed[:24]}"
        workspace_relpath = _canonical_session_workspace_relpath(
            investigation_session_id=investigation_session_id,
            research_task_id=context.research_task_id,
            investigation_context_id=context.investigation_context_id,
            hypothesis_execution_input_record_id=selected_hypothesis.scientific_record_id,
        )
        return InvestigationSessionIdentity(
            investigation_session_id=investigation_session_id,
            investigation_context_id=context.investigation_context_id,
            research_task_id=context.research_task_id,
            hypothesis_execution_input_record_id=selected_hypothesis.scientific_record_id,
            workspace_identity=workspace_relpath,
            minichem_scope_id=f"investigation-scope-{seed[24:48]}",
            scope_contract_digest=scope_contract,
        )

    def ensure_session(self, session: InvestigationSessionIdentity) -> None:
        """核对并持久化外部 scope owner；绝不提交作业。"""

        material_preparer = self.dependencies.get(
            "prepare_investigation_session_materials"
        )
        if material_preparer is not None:
            if not callable(material_preparer):
                raise TypeError(
                    "prepare_investigation_session_materials 必须可调用"
                )
            materialized = material_preparer(session)
            if materialized is not None:
                raise TypeError(
                    "prepare_investigation_session_materials 必须返回 None"
                )
        prepare = self.dependencies.get("prepare_investigation_session")
        if prepare is not None:
            if not callable(prepare):
                raise TypeError("prepare_investigation_session 必须可调用")
            # Context Store 已经持久化 Session 身份后，先完成确认后的目录冻结，
            # 再创建 scope。这样 9011 的 Catalog 快照成为本次 Investigation
            # 的执行依赖，且不会在候选/待确认阶段提前访问服务。
            prepared = prepare(session)
            if prepared is not None:
                raise TypeError("prepare_investigation_session 必须返回 None")
        checker = self.dependencies.get("ensure_investigation_session")
        if checker is not None:
            result = checker(session)
            if result is not None:
                raise ValueError("ensure_investigation_session 必须返回 None")
        manager = self.dependencies.get("native_scope_manager")
        if manager is not None:
            ensure = getattr(manager, "ensure_session", None)
            if not callable(ensure):
                raise TypeError("native_scope_manager 缺少 ensure_session")
            ensure(session)


class _RuntimeProviderTransport(NativeProviderTransport):
    """把已有 Codex runtime 暴露为 native backend transport。"""

    def __init__(self, runtime: InvestigationCodexRuntime) -> None:
        self.runtime = runtime

    def run_investigation_turn(self, request: Mapping[str, Any], **kwargs: Any) -> Any:
        """转发一次 Investigation turn；恢复身份由 backend 传入的参数核对。"""

        return self.runtime.run_investigation_turn(request, **kwargs)

    def replay_completed_turn(self, request: Mapping[str, Any], **kwargs: Any) -> Any:
        """只读重放同一 completed turn，供 Harness 补齐恢复事实。"""

        return self.runtime.replay_completed_turn(request, **kwargs)


class _MiniChemExecutionTransport(NativeExecutionTransport):
    """将 MiniChem open execution adapter 映射到 native backend。"""

    def __init__(
        self,
        adapter: MiniChemMcpAdapter,
    ) -> None:
        self.adapter = adapter
        self._opened = False
        self._submission_catalog_identity: dict[str, Any] | None = None

    def _ensure_open(self) -> None:
        if not self._opened:
            self.adapter.open()
            self._opened = True

    def _verify_submission_catalog(self) -> Mapping[str, Any]:
        """只在新提交前核对 9012 与冻结 Action/Backend 合同的目录身份。"""

        self._ensure_open()
        if self._submission_catalog_identity is not None:
            return self._submission_catalog_identity
        discovery = self.adapter.list_action_domains()
        catalog_hash = discovery.get("catalog_hash")
        catalog_version = discovery.get("catalog_version")
        if not isinstance(catalog_hash, str) or re.fullmatch(
            r"[0-9a-f]{64}", catalog_hash
        ) is None:
            raise RuntimeError("9012 discovery 缺少合法 catalog_hash")
        if (
            not isinstance(catalog_version, str)
            or not catalog_version.strip()
            or catalog_version != catalog_version.strip()
        ):
            raise RuntimeError("9012 discovery 缺少合法 catalog_version")
        self._submission_catalog_identity = {
            "catalog_hash": catalog_hash,
            "catalog_version": catalog_version,
        }
        return self._submission_catalog_identity

    def submit_job(self, action: NativeAction, *, session: SessionReference) -> Mapping[str, Any]:
        """提交一次 action；scope submission identity 由服务端返回 operation。"""

        self._verify_submission_catalog()
        payload = dict(action.payload)
        payload.pop("operation_id", None)
        submission_key = payload.get("submission_key")
        if not isinstance(submission_key, str) or not submission_key.strip():
            submission_key = f"investigation-{_digest({'investigation_session_id': session.investigation_session_id, 'action_id': action.action_id, 'payload': payload})[:48]}"
        payload["submission_key"] = submission_key
        if payload.get("job_family") == "programmable_analysis" or payload.get("backend") == "programmable_analysis":
            return self.adapter.submit_analysis_program(payload, submission_key=submission_key)
        return self.adapter.submit_native_job(payload, submission_key=submission_key)

    def preflight_scope_readiness(
        self,
        session: SessionReference,
    ) -> Mapping[str, Any]:
        """完成当前 scope 的 MCP initialize 与关键 Catalog 身份核对。"""

        del session
        discovery = self._verify_submission_catalog()
        return {
            "authenticated": True,
            "catalog_hash": discovery.get("catalog_hash"),
            "catalog_version": discovery.get("catalog_version"),
        }

    def query_job(self, job: Any, *, session: SessionReference) -> Mapping[str, Any]:
        """只读查询同一 job，不做 readiness 或新提交。"""

        self._ensure_open()
        return self.adapter.get_execution_job(job.minichem_job_id)

    def collect_job(self, job: Any, *, session: SessionReference) -> Mapping[str, Any]:
        """收集同一 job 的终态产物。"""

        self._ensure_open()
        return self.adapter.collect_execution_job(job.minichem_job_id, include_inputs=False)


class InvestigationSessionHarnessOwner:
    """按持久 Session 身份懒加载一个独立 Investigation Harness。"""

    def __init__(
        self,
        *,
        task_root: Path,
        working_memory: ResearchTaskWorkingMemory,
        task_input: Mapping[str, Any],
        policy: Mapping[str, Any],
        dependencies: Mapping[str, Any],
    ) -> None:
        self.task_root = Path(task_root)
        self.working_memory = working_memory
        self.task_input = copy.deepcopy(dict(task_input))
        self.policy = copy.deepcopy(dict(policy))
        self.dependencies = dict(dependencies)
        (
            self._investigation_compact_prompt,
            self._investigation_model_context_window,
        ) = _resolve_investigation_codex_thread_dependency(self.dependencies)
        self._harnesses: dict[str, InvestigationHarness] = {}
        self._session_identities: dict[str, InvestigationSessionIdentity] = {}
        self._material_descriptors_by_session: dict[
            str, tuple[dict[str, Any], ...]
        ] = {}
        material_store = self.dependencies.get("research_material_store")
        if material_store is not None and not isinstance(
            material_store, ResearchMaterialStore
        ):
            raise TypeError("research_material_store 必须是 ResearchMaterialStore")
        self._research_material_store = material_store
        self._catalog_lock = threading.RLock()
        self._frozen_catalog: FrozenMiniChemCatalog | None = None
        injected_catalog = self.dependencies.get("frozen_catalog")
        if injected_catalog is not None:
            if not isinstance(injected_catalog, FrozenMiniChemCatalog):
                raise TypeError("frozen_catalog 必须是 FrozenMiniChemCatalog")
            self._frozen_catalog = injected_catalog
        self._artifact_manifest_registry = self.dependencies.get(
            "artifact_manifest_registry"
        )
        if not isinstance(self._artifact_manifest_registry, dict):
            self._artifact_manifest_registry = {}
            self.dependencies["artifact_manifest_registry"] = self._artifact_manifest_registry

    def _ensure_frozen_catalog(self) -> FrozenMiniChemCatalog:
        """首次 native Session 前冻结或启动复验 9011 Catalog。"""

        with self._catalog_lock:
            if self._frozen_catalog is not None:
                return self._frozen_catalog
            catalog_root = self.task_root / "internal" / "catalog"
            candidates = sorted(
                path
                for path in catalog_root.iterdir()
                if path.is_dir() and re.fullmatch(r"[0-9a-f]{64}", path.name)
            ) if catalog_root.exists() else []
            if len(candidates) > 1:
                raise RuntimeError("native Investigation 发现多个冻结 Catalog")
            if candidates:
                self._frozen_catalog = load_frozen_minichem_catalog_startup(
                    task_root=self.task_root,
                    catalog_hash=candidates[0].name,
                )
                return self._frozen_catalog
            endpoint = str(
                self.dependencies.get(
                    "catalog_endpoint",
                    os.environ.get(
                        "MINICHEM_CATALOG_ENDPOINT",
                        "http://127.0.0.1:9011/mcp",
                    ),
                )
            )
            adapter = MiniChemMcpAdapter(
                endpoint=endpoint,
                operation_root=self.task_root / "artifacts" / "minichem_catalog_operations",
            )
            try:
                adapter.open()
                self._frozen_catalog = freeze_minichem_catalog(
                    adapter=adapter,
                    task_root=self.task_root,
                )
            finally:
                adapter.close()
            return self._frozen_catalog

    def prepare_session(self, session: InvestigationSessionIdentity) -> None:
        """在创建 scope 前完成确认后的 Catalog 冻结。

        Context Store 会先保存 Session 身份，再由 ``InvestigationSessionOwner`` 调用本方法。
        本方法只读取或冻结任务级 Catalog，不创建 scope、workspace、Codex thread、
        MiniChem job 或模型 turn；冻结失败会让当前 Investigation 保持可恢复的 Open
        状态。``open_session`` 仍只负责已经冻结依赖下的 Harness 装配。
        """

        if not isinstance(session, InvestigationSessionIdentity):
            raise TypeError("session 必须是 InvestigationSessionIdentity")
        try:
            self._ensure_frozen_catalog()
        except (MiniChemCatalogFreezeError, MiniChemMcpError, OSError) as exc:
            # Catalog 服务或冻结包暂时不可用时，Context backend 会把该异常收敛成
            # 可恢复的 Open Investigation；不创建 scope，也不伪造科学失败。
            raise InvestigationSessionTemporarilyUnavailable(
                "native Investigation Catalog 暂时不可用"
            ) from exc

    def _build_authenticated_gateway(
        self,
        session: InvestigationSessionIdentity,
        session_ref: SessionReference,
        recovery_store: Any,
    ) -> InvestigationMiniChemGateway:
        """为当前 scope 构造只读证据网关，并复用同一 Evidence store。"""

        manager = self.dependencies.get("native_scope_manager")
        if manager is None:
            raise RuntimeError("native Investigation 缺少 scope capability manager")
        capability = manager.capability_for(session)
        evidence_store = self.dependencies.get("review_evidence_store")
        if not isinstance(evidence_store, ReviewEvidenceStore):
            evidence_store = ReviewEvidenceStore(self.task_root / "artifacts" / "evidence")
            self.dependencies["review_evidence_store"] = evidence_store
        endpoint = _investigation_endpoint(self.dependencies)
        return InvestigationMiniChemGateway(
            endpoint=endpoint,
            scope_capabilities={session.minichem_scope_id: capability},
            evidence_store=evidence_store,
            source_cache_dir=self.task_root / "artifacts" / "evidence" / "source_cache",
        )

    def _capability_generation_digest(
        self,
        session: InvestigationSessionIdentity,
        manager: Any | None,
    ) -> str:
        """从 capability owner 读取真实 generation；绝不复用合同摘要。"""

        capability = None
        if manager is not None:
            capability = manager.capability_for(session)
        else:
            injected = self.dependencies.get("capability_generation_digest")
            if isinstance(injected, str) and re.fullmatch(r"[0-9a-f]{64}", injected):
                return injected
        if capability is not None:
            for field_name in (
                "capability_generation_digest",
                "generation_digest",
                "scope_credential_generation_digest",
            ):
                value = (
                    capability.get(field_name)
                    if isinstance(capability, Mapping)
                    else getattr(capability, field_name, None)
                )
                if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
                    return value
            # 测试/离线 owner 可能只暴露 token；以 owner 返回的 token 派生
            # 稳定核对摘要，仍不把 policy contract 冒充 generation。
            token = (
                capability.get("token")
                if isinstance(capability, Mapping)
                else getattr(capability, "token", None)
            )
            if isinstance(token, str) and token:
                return _digest(
                    {
                        "scope_id": session.minichem_scope_id,
                        "capability_token": token,
                    }
                )
        # 无外部 capability owner 的离线 transport 仍要有独立 generation
        # 身份；真实生产入口始终由 manager 提供上面的 owner digest。
        return _digest(
            {
                "scope_id": session.minichem_scope_id,
                "owner": "offline-native-transport",
            }
        )

    def _ensure_frozen_catalog_checker(self) -> _FrozenCatalogSubmissionContractChecker:
        """返回当前任务唯一冻结 Catalog 的局部提交合同检查器。"""

        return _FrozenCatalogSubmissionContractChecker(self._ensure_frozen_catalog())

    def _check_frozen_submission_contract(
        self,
        action: NativeAction,
        *,
        session: SessionReference,
    ) -> bool:
        """只在将要产生新副作用时加载冻结 Catalog 并核对 Action/Backend。"""

        return self._ensure_frozen_catalog_checker().check_submission_contract(
            action,
            session=session,
        )

    @staticmethod
    def _execution_input_field_names(
        record: ScientificRecord,
    ) -> tuple[str, str]:
        """按记录 schema 选择当前或历史 Confirmation 字段。"""

        if (
            record.scientific_record_schema_version
            == HYPOTHESIS_CONFIRMATION_V2_SCHEMA_VERSION
        ):
            return (
                "source_selected_hypothesis_record_id",
                "material_upload_refs",
            )
        return "source_selected_hypothesis_ref", "material_refs"

    @staticmethod
    def _require_record_payload_mapping(
        record: ScientificRecord,
        label: str,
    ) -> Mapping[str, Any]:
        """读取科学记录的公开 payload object，不展开任何正文或外部路径。"""

        payload = record.payload
        if not isinstance(payload, Mapping):
            raise ValueError(f"{label}.payload 必须是 JSON object")
        return payload

    def _read_execution_material_records(
        self,
        execution_input: ScientificRecord,
        records: Sequence[ScientificRecord],
    ) -> tuple[ScientificRecord, ...]:
        """按 material_id 解析确认输入的材料，并核对直接来源顺序。"""

        payload = self._require_record_payload_mapping(
            execution_input,
            "HypothesisExecutionInputRecord",
        )
        selected_field, material_field = self._execution_input_field_names(
            execution_input
        )
        material_refs = payload.get(material_field)
        if not isinstance(material_refs, (list, tuple)) or isinstance(
            material_refs, (str, bytes)
        ):
            raise ValueError("HypothesisExecutionInputRecord.material_refs 非法")
        if any(not isinstance(ref, str) or not ref for ref in material_refs):
            raise ValueError("HypothesisExecutionInputRecord.material_refs 非法")
        if len(set(material_refs)) != len(material_refs):
            raise ValueError("HypothesisExecutionInputRecord.material_refs 不能重复")

        materials_by_id: dict[str, ScientificRecord] = {}
        for record in records:
            if record.scientific_record_type != "ResearchMaterialRecord":
                continue
            material_payload = self._require_record_payload_mapping(
                record,
                "ResearchMaterialRecord",
            )
            material_id = material_payload.get("material_id")
            if not isinstance(material_id, str) or not material_id:
                raise ValueError("ResearchMaterialRecord.material_id 非法")
            if material_id in materials_by_id:
                raise ValueError("ResearchMaterialRecord.material_id 重复")
            materials_by_id[material_id] = record
        materials: list[ScientificRecord] = []
        for material_id in material_refs:
            material = materials_by_id.get(material_id)
            if material is None:
                raise ValueError("HypothesisExecutionInputRecord 材料缺失")
            materials.append(material)

        source_refs = tuple(execution_input.source_scientific_record_ids)
        selected_ref = payload.get(selected_field)
        if not isinstance(selected_ref, str) or not selected_ref:
            raise ValueError("HypothesisExecutionInputRecord 缺少 source_selected_hypothesis_ref")
        direct_selected = [
            record
            for record in records
            if record.scientific_record_id == selected_ref
            and record.scientific_record_type == "SelectedHypothesisRecord"
        ]
        direct_decisions = [
            record
            for record in records
            if record.scientific_record_id in source_refs
            and record.scientific_record_type == "HypothesisExecutionDecisionRecord"
        ]
        if len(direct_selected) != 1 or len(direct_decisions) != 1:
            raise ValueError("HypothesisExecutionInputRecord 来源缺少 Decision/Selected")
        records_by_ref = {record.scientific_record_id: record for record in records}
        direct_material_refs = tuple(
            ref
            for ref in source_refs
            if ref in records_by_ref
            and records_by_ref[ref].scientific_record_type == "ResearchMaterialRecord"
        )
        expected_refs = {
            direct_decisions[0].scientific_record_id,
            selected_ref,
            *(material.scientific_record_id for material in materials),
        }
        expected_material_refs = {
            material.scientific_record_id for material in materials
        }
        if (
            len(source_refs) != len(set(source_refs))
            or set(source_refs) != expected_refs
            or len(direct_material_refs) != len(expected_material_refs)
            or set(direct_material_refs) != expected_material_refs
        ):
            raise ValueError(
                "HypothesisExecutionInputRecord.source_scientific_record_ids 不匹配"
            )
        return tuple(materials)

    def _resolve_execution_material_descriptors(
        self,
        execution_input: ScientificRecord,
        records: Sequence[ScientificRecord],
        session: SessionReference,
    ) -> list[dict[str, Any]]:
        """通过注入 seam 获取已物化材料 descriptor，不读取材料字节。"""

        materials = self._read_execution_material_records(execution_input, records)
        if not materials:
            return []
        cached = self._material_descriptors_by_session.get(
            session.investigation_session_id
        )
        if cached is not None:
            if len(cached) != len(materials):
                raise ValueError("Session 已物化材料数量与执行输入不一致")
            return [dict(descriptor) for descriptor in cached]
        resolver = None
        for dependency_name in (
            "material_descriptor_resolver",
            "investigation_material_descriptor_resolver",
            "material_resolver",
            "research_material_resolver",
        ):
            candidate = self.dependencies.get(dependency_name)
            if candidate is not None:
                resolver = candidate
                break
        if resolver is None:
            raise ValueError("确认执行输入缺少材料 descriptor resolver")

        descriptors: list[dict[str, Any]] = []
        for material in materials:
            if callable(resolver):
                raw_descriptor = resolver(material, session)
            else:
                raw_descriptor = None
                for method_name in (
                    "resolve_material_descriptor",
                    "materialize_for_session",
                    "resolve",
                ):
                    method = getattr(resolver, method_name, None)
                    if callable(method):
                        raw_descriptor = method(material, session)
                        break
                if raw_descriptor is None:
                    raise TypeError("材料 resolver 必须提供 descriptor 方法")
            if not isinstance(raw_descriptor, Mapping):
                raise ValueError("材料 resolver 必须返回 descriptor object")
            material_payload = self._require_record_payload_mapping(
                material,
                "ResearchMaterialRecord",
            )
            descriptor = dict(raw_descriptor)
            # 物化 seam 可以只产生 workspace_relpath；其它模型可见字段始终取自
            # 已确认的 ResearchMaterialRecord，绝不从外部文件或 content_ref 读取。
            for field_name in (
                "filename",
                "media_type",
                "size_bytes",
                "sha256",
                "description",
            ):
                if field_name not in descriptor and field_name in material_payload:
                    descriptor[field_name] = material_payload[field_name]
            descriptors.append(descriptor)
        return descriptors

    def prepare_session_materials(
        self,
        session: InvestigationSessionIdentity,
    ) -> None:
        """创建 Session workspace，并在 scope/provider 前物化已确认材料。

        输入是已经写入 Context Store 的 Session 身份；动作只读取该 Session 的
        ``HypothesisExecutionInputRecord`` 闭包，并通过任务级材料 store 核对和
        复制声明的字节。返回无值。材料缺失或摘要漂移会由 store 抛出 typed 异常，
        调用方不得继续创建 scope 或启动 provider。
        """

        if not isinstance(session, InvestigationSessionIdentity):
            raise TypeError("session 必须是 InvestigationSessionIdentity")
        workspace = _session_owner_workspace(self.task_root, session)
        workspace.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(workspace, 0o700)
        records = self._read_record_closure(
            session.research_task_id,
            (session.hypothesis_execution_input_record_id,),
        )
        execution_inputs = [
            record
            for record in records
            if record.scientific_record_id == session.hypothesis_execution_input_record_id
            and record.scientific_record_type == "HypothesisExecutionInputRecord"
        ]
        if not execution_inputs:
            self._session_identities[session.investigation_session_id] = session
            self._material_descriptors_by_session[
                session.investigation_session_id
            ] = ()
            return
        if len(execution_inputs) != 1:
            raise ValueError("Session 无法确定唯一 HypothesisExecutionInputRecord")
        execution_input = execution_inputs[0]
        materials = self._read_execution_material_records(execution_input, records)
        if materials and self._research_material_store is None:
            raise ValueError("确认执行输入缺少任务级研究材料存储")
        descriptors: list[dict[str, Any]] = []
        for material in materials:
            payload = self._require_record_payload_mapping(
                material,
                "ResearchMaterialRecord",
            )
            descriptor = {
                "filename": payload["filename"],
                "media_type": payload["media_type"],
                "size_bytes": payload["size_bytes"],
                "sha256": payload["sha256"],
                "content_ref": payload["content_ref"],
            }
            if payload.get("description") is not None:
                descriptor["description"] = payload["description"]
            relative_path = self._research_material_store.materialize_material(
                descriptor,
                workspace,
            )
            descriptors.append(
                {
                    "workspace_relpath": relative_path,
                    "filename": payload["filename"],
                    "media_type": payload["media_type"],
                    "size_bytes": payload["size_bytes"],
                    "sha256": payload["sha256"],
                    "description": payload.get("description"),
                }
            )
        self._session_identities[session.investigation_session_id] = session
        self._material_descriptors_by_session[
            session.investigation_session_id
        ] = tuple(descriptors)

    def _build_provider_request(self, request: InvestigationHarnessRequest, *, session: SessionReference, action_results: Sequence[Mapping[str, Any]] = (), is_resume: bool = False, thread_id: str | None = None) -> Mapping[str, Any]:
        """从 V2 Selected 或确认执行输入/Repair records 构造模型 prompt。"""

        from chemistry_multiagent.investigation.prompts import (
            build_initial_investigation_request,
            build_repair_investigation_request,
        )

        records = self._read_record_closure(
            request.research_task_id, request.input_scientific_record_ids
        )
        execution_inputs = [
            record
            for record in records
            if record.scientific_record_type == "HypothesisExecutionInputRecord"
        ]
        if len(execution_inputs) > 1:
            raise ValueError("Investigation prompt 找到多个执行输入")
        execution_input = execution_inputs[0] if execution_inputs else None
        selected = next(
            (r for r in records if r.scientific_record_type == "SelectedHypothesisRecord"),
            None,
        )
        records_by_id = {record.scientific_record_id: record for record in records}
        direct_repair_requests = [
            records_by_id[record_id]
            for record_id in request.input_scientific_record_ids
            if records_by_id[record_id].scientific_record_type == "RepairRequestRecord"
        ]
        if len(direct_repair_requests) > 1:
            raise ValueError("Investigation prompt 找到多个直接 RepairRequestRecord")
        repair = direct_repair_requests[0] if direct_repair_requests else None
        question = self.task_input.get("scientific_question")
        model = os.environ.get("ARCHE_CODEX_MODEL_NAME", "gpt-5.6-sol")
        validation_results = tuple(
            dict(item)
            for item in action_results
            if item.get("kind") == "result_validation"
        )
        if validation_results:
            if len(validation_results) != 1 or len(validation_results) != len(action_results):
                raise ValueError("Evidence Result 修正一次只接受一个验证结果")
            if thread_id is None:
                thread_id = self._load_existing_thread_id(request, session)
            if thread_id is None:
                raise ValueError("Evidence Result 修正缺少既有 Codex thread")
            result_version = (
                str(repair.payload.get("result_version", "v1"))
                if repair is not None
                else "v1"
            )
            return build_repair_investigation_request(
                request.research_task_id,
                session.investigation_context_id,
                session.hypothesis_execution_input_record_id,
                thread_id,
                result_version,
                json.dumps(
                    validation_results[0],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                confirmed_execution_input=execution_input is not None,
                model=model,
            )
        if repair is not None:
            result_version = str(repair.payload.get("result_version", "v1"))
            if thread_id is None:
                thread_id = self._load_existing_thread_id(request, session)
            if thread_id is None:
                raise ValueError("Repair request 缺少既有 Codex thread")
            return build_repair_investigation_request(
                request.research_task_id,
                session.investigation_context_id,
                session.hypothesis_execution_input_record_id,
                thread_id,
                result_version,
                str(
                    repair.payload.get(
                        "repair_advice",
                        repair.payload.get("instructions", "继续验证"),
                    )
                ),
                confirmed_execution_input=execution_input is not None,
                model=model,
            )
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Investigation prompt 缺少冻结问题")
        if execution_input is not None:
            execution_payload = self._require_record_payload_mapping(
                execution_input,
                "HypothesisExecutionInputRecord",
            )
            selected_field, material_field = self._execution_input_field_names(
                execution_input
            )
            if set(execution_payload) != {
                selected_field,
                "execution_rank",
                "hypothesis",
                "guidance",
                material_field,
            }:
                raise ValueError("HypothesisExecutionInputRecord payload 合同非法")
            hypothesis = execution_payload["hypothesis"]
            if not isinstance(hypothesis, Mapping):
                raise ValueError("HypothesisExecutionInputRecord.hypothesis 非法")
            guidance = execution_payload["guidance"]
            if guidance is not None and (
                not isinstance(guidance, str) or not guidance.strip()
            ):
                raise ValueError("HypothesisExecutionInputRecord.guidance 非法")
            materials = self._resolve_execution_material_descriptors(
                execution_input,
                records,
                session,
            )
            return build_initial_investigation_request(
                request.research_task_id,
                session.investigation_context_id,
                session.hypothesis_execution_input_record_id,
                question,
                hypothesis,
                guidance=guidance,
                materials=materials,
                confirmed_execution_input=True,
                codex_thread_id=thread_id,
                model=model,
            )
        if selected is None:
            raise ValueError("Investigation prompt 缺少冻结假设")
        selected_payload = self._require_record_payload_mapping(
            selected,
            "SelectedHypothesisRecord",
        )
        hypothesis = selected_payload.get("hypothesis")
        if not isinstance(hypothesis, Mapping):
            raise ValueError("SelectedHypothesisRecord.hypothesis 非法")
        context = self.task_input.get("supplementary_scientific_input", {})
        if not isinstance(context, Mapping):
            context = {}
        return build_initial_investigation_request(
            request.research_task_id,
            session.investigation_context_id,
            session.hypothesis_execution_input_record_id,
            question,
            hypothesis,
            frozen_context=context,
            explicit_conditions=context,
            codex_thread_id=thread_id,
            model=model,
        )

    def _load_existing_thread_id(
        self,
        request: InvestigationHarnessRequest,
        session: SessionReference,
    ) -> str | None:
        """从同一 Session 的 Codex manifest 恢复跨 Module Invocation 的 thread。

        首次 Investigation 与后续 Repair 是不同的 Module Invocation，但必须共享
        同一 Session owner。thread 不属于 Working Memory 科学事实；它由
        CodexSessionStore 按其底层 owner 协议持久化。找不到 manifest 时返回
        ``None``，调用方保留为可恢复的证据缺失，而不是猜测新 thread。
        """

        model = os.environ.get("ARCHE_CODEX_MODEL_NAME", "gpt-5.6-sol")
        root = (
            self.task_root
            / "internal"
            / "state"
            / "investigation_runtime"
            / "sessions"
            / session.investigation_session_id
        )
        owner = CodexSessionOwner.investigation(
            request.research_task_id,
            session.investigation_context_id,
            session.hypothesis_execution_input_record_id,
        )
        store = CodexSessionStore.from_root(
            root / "codex_sessions",
            owner=owner,
            scope_id=session.minichem_scope_id,
            model=model,
        )
        manifest = store.load()
        value = None if manifest is None else manifest.get("thread_id")
        return value if isinstance(value, str) and value.strip() else None

    def _read_record_closure(
        self,
        research_task_id: str,
        scientific_record_ids: Sequence[str],
    ) -> tuple[ScientificRecord, ...]:
        """读取当前 Harness 请求引用的完整科学来源闭包。

        首次 Investigation 的直接输入通常就是 SelectedHypothesisRecord；返修输入
        则是 RepairRequestRecord。两者都必须沿 source_scientific_record_ids 回溯
        到同一个 SelectedHypothesisRecord，不能用列表位置或旧运行状态猜测。
        """

        pending = list(scientific_record_ids)
        records: dict[str, ScientificRecord] = {}
        while pending:
            record_id = pending.pop()
            if record_id in records:
                continue
            values = self.working_memory.read_research_task_scientific_records(
                research_task_id, (record_id,)
            )
            if len(values) != 1:
                raise ValueError("Scientific Record 闭包不完整")
            record = values[0]
            if (
                record.research_task_id != research_task_id
                or record.scientific_record_id != record_id
            ):
                raise ValueError("Scientific Record 身份漂移")
            records[record_id] = record
            pending.extend(
                source_ref
                for source_ref in record.source_scientific_record_ids
                if source_ref not in records
            )
        return tuple(records.values())

    def _runtime(self, session: InvestigationSessionIdentity) -> InvestigationCodexRuntime:
        factory = self.dependencies.get("investigation_runtime_factory")
        if callable(factory):
            runtime = factory(session)
            if runtime is None:
                raise RuntimeError("investigation_runtime_factory 返回空对象")
            return runtime
        root = (
            self.task_root
            / "internal"
            / "state"
            / "investigation_runtime"
            / "sessions"
            / session.investigation_session_id
        )
        # CodexSessionStore.from_root() 会按 Investigation owner 派生实际的
        # workspace/CODEX_HOME。sandbox 必须绑定到同一对目录；若使用 session 根目录，
        # runtime 会在真正启动 app-server 前拒绝“workspace does not match turn store”。
        workspace = _session_owner_workspace(self.task_root, session)
        owner_root = workspace.parent
        codex_home = owner_root / "codex_home"
        runtime_root = root / "codex_runtime"
        for path in (workspace, codex_home, runtime_root):
            path.mkdir(parents=True, exist_ok=True)
        codex_bin = self.dependencies.get("codex_bin", os.environ.get("ARCHE_CODEX_BIN", "codex"))
        endpoint = _investigation_endpoint(self.dependencies)
        environment = dict(os.environ)
        manager = self.dependencies.get("native_scope_manager")
        if manager is not None:
            capability = manager.capability_for(session)
            token_env = str(
                self.dependencies.get(
                    "minichem_capability_token_env",
                    "ARCHE_MINICHEM_CAPABILITY_TOKEN",
                )
            )
            environment[token_env] = capability.token
        sandbox = CodexProcessSandboxSpec(workspace=workspace, codex_home=codex_home, codex_bin=Path(codex_bin), runtime_root=runtime_root)
        return InvestigationCodexRuntime(
            session_root=root / "codex_sessions",
            trace_root=root / "traces",
            minichem=InvestigationMcpSettings(endpoint, str(self.dependencies.get("minichem_capability_token_env", "ARCHE_MINICHEM_CAPABILITY_TOKEN"))),
            codex_bin=codex_bin,
            investigation_compact_prompt=self._investigation_compact_prompt,
            investigation_model_context_window=(
                self._investigation_model_context_window
            ),
            environment=environment,
            adapter=self.dependencies.get("codex_adapter"),
            sandbox_spec=sandbox,
            role="investigation",
        )

    def open_session(self, session: InvestigationSessionIdentity) -> InvestigationHarness:
        """打开或重用同一 Session 的 Harness；不创建第二份恢复事实。"""

        existing = self._harnesses.get(session.investigation_session_id)
        if existing is not None:
            return existing
        self._session_identities[session.investigation_session_id] = session
        root = (
            self.task_root
            / "internal"
            / "state"
            / "investigation_runtime"
            / "sessions"
            / session.investigation_session_id
        )
        recovery_store_root = root / "harness"
        manager = self.dependencies.get("native_scope_manager")
        if manager is not None:
            manager.ensure_session(session)
        session_ref = SessionReference(
            research_task_id=session.research_task_id,
            investigation_context_id=session.investigation_context_id,
            hypothesis_execution_input_record_id=(
                session.hypothesis_execution_input_record_id
            ),
            investigation_session_id=session.investigation_session_id,
            workspace_identity=session.workspace_identity,
            minichem_scope_id=session.minichem_scope_id,
            scope_contract_digest=session.scope_contract_digest,
            capability_generation_digest=self._capability_generation_digest(
                session, manager
            ),
        )
        provider = self.dependencies.get("native_provider_transport")
        provider_factory = self.dependencies.get("native_provider_transport_factory")
        runtime: InvestigationCodexRuntime | None = None
        if provider is None:
            runtime = self._runtime(session)
        if callable(provider_factory):
            provider = provider_factory(session, runtime)
        if provider is None:
            if runtime is None:
                runtime = self._runtime(session)
            provider = _RuntimeProviderTransport(runtime)
        execution = self.dependencies.get("native_execution_transport")
        if callable(self.dependencies.get("native_execution_transport_factory")):
            execution = self.dependencies["native_execution_transport_factory"](session)
        if execution is None:
            if manager is None:
                raise RuntimeError("native Investigation 默认入口必须配置 9012 scope registry")
            capability = manager.capability_for(session)
            endpoint = _investigation_endpoint(self.dependencies)
            adapter_factory = self.dependencies.get("native_minichem_adapter_factory")
            if adapter_factory is None:
                adapter_factory = self.dependencies.get("minichem_adapter_factory")
            adapter_kwargs = {
                "endpoint": endpoint,
                "session_factory": lambda: _AuthenticatedStreamableHTTPContext(
                    endpoint,
                    {"X-MiniChem-Capability": capability.token},
                ),
                "operation_root": root / "minichem_operations",
            }
            if callable(adapter_factory):
                adapter = adapter_factory(**adapter_kwargs)
            else:
                adapter = MiniChemMcpAdapter(**adapter_kwargs)
            execution = _MiniChemExecutionTransport(adapter)
        from chemistry_multiagent.investigation.harness_recovery_store import HarnessRecoveryStore

        store = HarnessRecoveryStore(recovery_store_root / "recovery.sqlite3")
        contract_checker = self.dependencies.get("submission_contract_checker")
        if contract_checker is None and manager is not None:
            contract_checker = self._check_frozen_submission_contract
        backend = InvestigationSessionRunner(
            recovery_store=store,
            provider_transport=provider,
            execution_transport=execution,
            evidence_verifier=self.dependencies.get("evidence_verifier"),
            submission_contract_checker=contract_checker,
            require_evidence_verifier=manager is not None,
            provider_request_builder=self._build_provider_request,
            session_binder=lambda request, **_: session_ref,
        )
        if self.dependencies.get("evidence_verifier") is None and manager is not None:
            gateway = self._build_authenticated_gateway(session, session_ref, store)
            backend._evidence_verifier = _NativeEvidenceVerifier(
                gateway=gateway,
                session=session_ref,
                recovery_store=store,
                artifact_manifest_registry=self._artifact_manifest_registry,
            )
        harness = InvestigationHarness(recovery_store_root, backend=backend, recovery_store=store)
        self._harnesses[session.investigation_session_id] = harness
        return harness


class InvestigationContextModuleAdapter(HarnessModuleRunner):
    """公开 Investigation module，内部由 Context + child Harness 完成。"""

    def __init__(self, harness: InvestigationHarness) -> None:
        super().__init__(harness)


def build_native_investigation_module_adapter(
    *,
    task_root: Path,
    working_memory: ResearchTaskWorkingMemory,
    task_input: Mapping[str, Any],
    policy: Mapping[str, Any],
    dependencies: Mapping[str, Any],
    research_material_store: ResearchMaterialStore | None = None,
    max_parallel_investigations: int = 5,
) -> InvestigationContextModuleAdapter:
    """构造 V2-native Investigation Context，不读取 schema-v10 事实。"""

    # 只有真实默认 transport 才需要接入 registry；确定性测试通过显式 fake
    # transport 时不触碰外部凭据或服务。
    if not isinstance(dependencies, dict):
        dependencies = dict(dependencies)
    if (
        dependencies.get("native_scope_manager") is None
        and dependencies.get("native_execution_transport") is None
        and not callable(dependencies.get("native_execution_transport_factory"))
    ):
        registry_root = os.environ.get("MINICHEM_SCOPE_REGISTRY_ROOT")
        if registry_root:
            dependencies["native_scope_manager"] = _NativeScopeCapabilityManager(
                task_root=Path(task_root)
            )
        else:
            dependencies["native_scope_manager"] = None
    if not isinstance(dependencies.get("artifact_manifest_registry"), dict):
        dependencies["artifact_manifest_registry"] = {}
    if research_material_store is not None:
        if not isinstance(research_material_store, ResearchMaterialStore):
            raise TypeError(
                "research_material_store 必须是 ResearchMaterialStore"
            )
        dependencies["research_material_store"] = research_material_store

    factory = InvestigationSessionHarnessOwner(
        task_root=Path(task_root),
        working_memory=working_memory,
        task_input=task_input,
        policy=policy,
        dependencies=dependencies,
    )
    if research_material_store is not None:
        dependencies["prepare_investigation_session_materials"] = (
            factory.prepare_session_materials
        )
    # 只有默认 native scope owner 才需要在 Session 外部资源创建前冻结服务目录。
    # 纯离线替代 transport 不配置 scope owner，本身不会进入真实 Investigation
    # 执行路径；保留其零服务装配能力供确定性测试使用。
    if dependencies.get("native_scope_manager") is not None:
        dependencies["prepare_investigation_session"] = factory.prepare_session
    context_store = InvestigationContextStore(
        Path(task_root) / "internal" / "state" / "investigation_context.sqlite3"
    )
    resolver = InvestigationSessionOwner(task_root=Path(task_root), policy=policy, dependencies=dependencies)
    backend = InvestigationContextRunner(
        context_store=context_store,
        working_memory=working_memory,
        session_owner=resolver,
        session_harness_owner=factory,
        max_parallel_investigations=max_parallel_investigations,
    )
    from chemistry_multiagent.investigation.harness_recovery_store import HarnessRecoveryStore

    store = HarnessRecoveryStore(
        Path(task_root)
        / "internal"
        / "state"
        / "investigation_context_harness.sqlite3"
    )
    return InvestigationContextModuleAdapter(
        InvestigationHarness(
            Path(task_root)
            / "internal"
            / "state"
            / "investigation_context_harness",
            backend=backend,
            recovery_store=store,
        )
    )


def build_native_review_module_adapter(
    *,
    task_root: Path,
    working_memory: ResearchTaskWorkingMemory,
    task_input: Mapping[str, Any],
    policy: Mapping[str, Any],
    dependencies: Mapping[str, Any],
) -> V2TaskReviewModuleAdapter:
    """构造只读 V2 Review adapter。

    测试或嵌入式调用可以注入 ``review_runtime`` 和 snapshot builder；生产默认则
    在当前 Run Bundle 下创建独立的 Evidence store 与 task-level Codex runtime。
    这里只创建目录和对象，不启动 Codex、连接 MCP 或读取证据正文。
    """

    if not isinstance(dependencies, dict):
        dependencies = dict(dependencies)
    runtime = dependencies.get("v2_review_runtime")
    if isinstance(runtime, ResearchTaskReviewRunner):
        return V2TaskReviewModuleAdapter(runtime)
    provider = dependencies.get("review_runtime")
    snapshot_builder = dependencies.get("evidence_snapshot_builder")
    evidence_lease = dependencies.get("evidence_lease")
    evidence_store = dependencies.get("review_evidence_store")
    evidence_token: str | None = None
    if provider is None:
        root = (
            Path(task_root)
            / "internal"
            / "state"
            / "investigation_runtime"
            / "review"
        )
        research_task_id = (
            dependencies.get("research_task_id")
            or task_input.get("research_task_id")
            or policy.get("research_task_id")
            or Path(task_root).name
        )
        review_owner = CodexSessionOwner.task_review(str(research_task_id))
        review_owner_root = (
            root
            / "codex_sessions"
            / "task_reviews"
            / str(research_task_id)
        )
        workspace = review_owner_root / "workspace"
        codex_home = review_owner_root / "codex_home"
        runtime_root = root / "codex_runtime"
        for path in (workspace, codex_home, runtime_root):
            path.mkdir(parents=True, exist_ok=True)
        codex_bin = dependencies.get(
            "review_codex_bin",
            dependencies.get("codex_bin", os.environ.get("ARCHE_CODEX_BIN", "codex")),
        )
        evidence_endpoint = dependencies.get(
            "evidence_mcp_endpoint",
            os.environ.get("EVIDENCE_MCP_ENDPOINT", "http://127.0.0.1:9020/mcp"),
        )
        evidence_header_env = dependencies.get(
            "evidence_capability_token_env",
            "ARCHE_EVIDENCE_CAPABILITY_TOKEN",
        )
        evidence_header = dependencies.get(
            "evidence_capability_header",
            DEFAULT_EVIDENCE_CAPABILITY_HEADER,
        )
        if evidence_store is None:
            evidence_store = ReviewEvidenceStore(
                Path(task_root) / "artifacts" / "evidence"
            )
            dependencies["review_evidence_store"] = evidence_store
        if evidence_lease is None:
            evidence_token_factory = dependencies.get("evidence_token_factory")
            if callable(evidence_token_factory):
                evidence_token = evidence_token_factory()
            else:
                evidence_token = secrets.token_urlsafe(32)
            evidence_lease = TaskEvidenceMcpLease(
                str(research_task_id),
                evidence_store.root_dir,
                dependencies.get(
                    "evidence_runtime_root",
                    Path(task_root).parent
                    / ".investigation_evidence_mcp"
                    / str(research_task_id),
                ),
                port=dependencies.get("evidence_mcp_port", 9020),
                capability_token=evidence_token,
                capability_header=str(evidence_header),
            )
        else:
            # 允许生产装配复用显式 lease；其 token 只短暂注入当前
            # Codex app-server 环境，不进入 Review request 或 Working Memory。
            candidate_token = dependencies.get("evidence_capability_token")
            if not isinstance(candidate_token, str) or not candidate_token:
                candidate_token = getattr(evidence_lease, "_capability_token", None)
            if isinstance(candidate_token, str) and candidate_token:
                evidence_token = candidate_token
        evidence_mcp = EvidenceMcpSettings(
            str(getattr(evidence_lease, "endpoint", evidence_endpoint)),
            {str(evidence_header): str(evidence_header_env)},
        )
        review_environment = dict(os.environ)
        if evidence_token is not None:
            for environment_name in evidence_mcp.env_http_headers.values():
                review_environment[environment_name] = evidence_token
        provider = InvestigationCodexRuntime(
            session_root=root / "codex_sessions",
            trace_root=root / "traces",
            evidence_mcp=evidence_mcp,
            codex_bin=codex_bin,
            environment=review_environment,
            adapter=dependencies.get("review_codex_adapter"),
            sandbox_spec=CodexProcessSandboxSpec(
                workspace=workspace,
                codex_home=codex_home,
                codex_bin=Path(codex_bin),
                runtime_root=runtime_root,
            ),
            role="task_review",
        )
    if not callable(snapshot_builder):
        if evidence_store is None:
            evidence_store = ReviewEvidenceStore(
                Path(task_root) / "artifacts" / "evidence"
            )
            if isinstance(dependencies, dict):
                dependencies["review_evidence_store"] = evidence_store
        artifact_manifest_provider = dependencies.get("artifact_manifest_provider")
        if artifact_manifest_provider is None:
            registry = dependencies.get("artifact_manifest_registry")
            # Investigation assembly 总会创建进程内 registry。重启后的空 registry
            # 不含任何恢复事实，必须回退到持久 Evidence store；只有确有内容时才
            # 把它作为同进程只读快路径。
            if not isinstance(registry, Mapping) or registry:
                artifact_manifest_provider = registry
        snapshot_builder = build_v2_evidence_snapshot_builder(
            evidence_store,
            artifact_manifest_provider,
        )
    runtime = ResearchTaskReviewRunner(
        working_memory=working_memory,
        review_runtime=provider,
        review_scope_id=str(policy.get("review_scope_id", "task-review")),
        evidence_snapshot_builder=snapshot_builder,
        evidence_reader_tool=dependencies.get("evidence_reader_tool"),
        evidence_lease=evidence_lease,
    )
    return V2TaskReviewModuleAdapter(runtime)


__all__ = [
    "InvestigationContextModuleAdapter",
    "InvestigationSessionHarnessOwner",
    "InvestigationSessionOwner",
    "build_native_investigation_module_adapter",
    "build_native_review_module_adapter",
]
