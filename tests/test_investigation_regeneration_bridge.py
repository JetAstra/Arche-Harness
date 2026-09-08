"""Investigation 假设重生 bridge 的屏障、历史和恢复测试。"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from chemistry_multiagent.agents.hypothesis_agent import HypothesisAgent
from chemistry_multiagent.investigation import (
    HypothesisRegenerationBridgeError,
    HypothesisRegenerationGenerationConfig,
    HypothesisRegenerationSessionIdentity,
    InvestigationRevisionConflict,
    InvestigationHypothesisRegenerationBridge,
    InvestigationSQLiteStore,
)
from chemistry_multiagent.investigation.regeneration_bridge import (
    HypothesisAgentRegenerationGenerator,
    _compose_batch_payload,
    _compose_initial_batch_payload,
    _fallback_single_query_provenance,
    _grouped_query_provenance,
    _compose_legacy_combination_outcome,
    _compose_legacy_batch_outcomes,
    _index_legacy_batch_records,
    _manifest_source_digest,
    _normalize_hypothesis_phase,
    _normalize_ranked_hypotheses,
    _read_regeneration_gate_marker_ids,
    _read_cleanup_plan_header,
    _read_initial_query_and_retrieval,
    _validate_receipt_cleanup_fields,
    _finding_by_combination,
    _validate_receipt_identity_fields,
    _validate_session_identities,
    _validate_task_root_against_task_created,
)
from chemistry_multiagent.investigation import regeneration_bridge as bridge_module
from chemistry_multiagent.utils.model_operation import ModelOperationError

from .test_investigation_regeneration_memory import (
    begin_next_batch,
    exhaust_three_repairs,
    make_initial_state,
    make_state_with_closed_history,
    mark_all_refuted,
)
from .test_investigation_working_memory import _make_reader
from .test_hypothesis_batches import (
    _catalog_provider,
    _hypothesis_arguments,
    _long_query,
    _regeneration_package,
    _ticket03_chemical_capability_projection,
)


def _configure_reader(reader: object, *, task_root: Path | None = None) -> object:
    """为 Working Memory fixture 补齐真实 TaskCreated 的 Hypothesis 预算。"""

    task_created = reader.task_created  # type: ignore[attr-defined]
    task_created["task_config"] = {
        "num_queries": 1,
        "num_hypotheses_per_query": 5,
        "top_n_hypotheses": 5,
        "max_parallel_investigations": 5,
        "max_investigation_repairs_per_combination": 3,
        "max_hypothesis_batches": 3,
    }
    bound_root = task_root or Path("local_data/runtime/bridge-test")
    task_created["task_output_relative_dir"] = (
        bound_root.relative_to(bound_root.parent).as_posix()
        if task_root is not None
        else "bridge-test"
    )
    reader.scratch_root = str(bound_root.parent.resolve())
    reader.investigation_sessions = {}

    def list_investigation_sessions(task_id, batch_id, *, include_internal=False):
        del task_id, include_internal
        return copy.deepcopy(reader.investigation_sessions.get(batch_id, []))

    reader.list_investigation_sessions = list_investigation_sessions
    return reader


def _generation_config() -> HypothesisRegenerationGenerationConfig:
    """构造与 fixture TaskCreated 一致的五组合生成配置。"""

    return HypothesisRegenerationGenerationConfig(
        task_root="local_data/runtime/bridge-test",
        stage_tool_projection=(),
        chemical_capability_projection=tuple(
            copy.deepcopy(_ticket03_chemical_capability_projection())
        ),
        num_queries=1,
        num_hypotheses_per_query=5,
        hypothesis_ids_by_query=(tuple(f"new-hypothesis-{index}" for index in range(1, 6)),),
        merge_hypothesis_ids=(),
    )


def _identities() -> tuple[HypothesisRegenerationSessionIdentity, ...]:
    """构造调用方显式提供的下一批五个 scope/workspace 身份。"""

    return tuple(
        HypothesisRegenerationSessionIdentity(
            combination_index=index,
            combination_id=f"new-combination-{index}",
            workspace_ref=f"workspace/new/{index}",
            scope_id=f"scope-new-{index}",
        )
        for index in range(1, 6)
    )


class _RecordingGenerator:
    """返回五项规范候选并记录完整 bridge 输入。"""

    def __init__(self) -> None:
        self.calls = []
        self.recovery_calls = []
        self.recovery_result = None

    @staticmethod
    def _result(task_root: str, operation_instance_id: str):
        """构造固定五项结果，供正常生成和 started replay 共用。"""

        ranked = [
            {
                "hypothesis_id": f"new-hypothesis-{index}",
                "strategy_name": f"新策略 {index}",
                "reasoning": f"新策略 {index} 的验证理由。",
                "rank": index,
            }
            for index in range(1, 6)
        ]
        base_input = {}
        base_encoded = json.dumps(
            base_input,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        base_digest = hashlib.sha256(base_encoded).hexdigest()
        base_path = (
            Path(task_root)
            / "working_memory"
            / "hypothesis_generation_inputs"
            / operation_instance_id
            / "base_input.json"
        )
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_path.write_bytes(base_encoded)
        manifest = {
            "schema_version": 1,
            "operation": "hypothesis_generation",
            "operation_instance_id": operation_instance_id,
            "base_input_ref": {
                "relative_path": base_path.relative_to(Path(task_root)).as_posix(),
                "sha256": base_digest,
                "media_type": "application/json",
                "size": len(base_encoded),
            },
            "prior_manifest_ref": None,
            "stages": [],
        }
        encoded = json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        manifest_path = (
            Path(task_root)
            / "working_memory"
            / "hypothesis_generation_inputs"
            / operation_instance_id
            / f"manifest_{digest}.json"
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_bytes(encoded)
        return {
            "hypothesis_result": {
                "queries": ["固定角度"],
                "hypotheses_by_query": [
                    {"query": "固定角度", "hypotheses": ranked},
                ],
                "ranked_strategies": ranked,
            },
            "operation_result": {
                "status": "validated",
                "manifest_ref": {
                    "relative_path": manifest_path.relative_to(Path(task_root)).as_posix(),
                    "sha256": digest,
                    "media_type": "application/json",
                    "size": len(encoded),
                },
            },
        }

    def generate_hypothesis_regeneration(self, request):
        self.calls.append(request)
        return self._result(
            request.generation_config.task_root,
            request.operation_instance_id,
        )

    def recover_hypothesis_regeneration(self, request, receipt):
        self.recovery_calls.append((request, receipt))
        return self.recovery_result or self._result(
            request.generation_config.task_root,
            request.operation_instance_id,
        )


class _BlockingGenerator(_RecordingGenerator):
    """在模型阶段阻塞，供并发 receipt claim 测试观察重复调用。"""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def generate_hypothesis_regeneration(self, request):
        self.calls.append(request)
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("测试模型等待超时")
        return self._result(
            request.generation_config.task_root,
            request.operation_instance_id,
        )


def test_top5_payload_composition_accepts_empty_session_identities(
    tmp_path: Path,
) -> None:
    """Top5 首批与重生 payload 都不要求 Investigation session。"""

    operation_id = "operation-top5"
    hypothesis_result = _RecordingGenerator._result(
        str(tmp_path),
        operation_id,
    )["hypothesis_result"]
    generation_config = replace(
        _generation_config(),
        task_root=str(tmp_path),
        creates_investigation_sessions=False,
    )

    initial_payload = _compose_initial_batch_payload(
        task_id="task-1",
        batch_id="batch-top5-1",
        operation_instance_id=operation_id,
        scientific_question="固定科学问题",
        initial_retrieval_phase={},
        hypothesis_result=hypothesis_result,
        generation_config=generation_config,
        identities=(),
    )
    regenerated_payload = _compose_batch_payload(
        state=make_initial_state("batch-top5-1"),
        next_batch_id="batch-top5-2",
        next_batch_index=2,
        operation_instance_id=operation_id,
        hypothesis_result=hypothesis_result,
        generation_config=generation_config,
        expected_queries=("固定角度",),
        causal_source_ref=None,
        source_record_ids=(),
        identities=(),
    )

    for payload in (initial_payload, regenerated_payload):
        selected = [
            item
            for item in payload["candidates"]
            if item["selection_status"] == "selected"
        ]
        assert len(selected) == 5
        assert [item["combination_index"] for item in selected] == [1, 2, 3, 4, 5]
        assert all(
            str(item["combination_id"]).startswith("top5-combination:")
            for item in selected
        )


class _ManifestAttackGenerator(_RecordingGenerator):
    """返回带指定清单破坏的结果，验证 bridge 不信任模型返回的路径。"""

    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode

    def generate_hypothesis_regeneration(self, request):
        self.calls.append(request)
        result = self._result(
            request.generation_config.task_root,
            request.operation_instance_id,
        )
        manifest_ref = result["operation_result"]["manifest_ref"]
        task_root = Path(request.generation_config.task_root)
        operation_dir = (
            task_root
            / "working_memory"
            / "hypothesis_generation_inputs"
            / request.operation_instance_id
        )
        if self.mode == "traversal":
            manifest_ref["relative_path"] = (
                "working_memory/hypothesis_generation_inputs/"
                f"{request.operation_instance_id}/../other/manifest.json"
            )
        elif self.mode == "cross_operation":
            manifest_ref["relative_path"] = (
                "working_memory/hypothesis_generation_inputs/other-operation/"
                + Path(manifest_ref["relative_path"]).name
            )
        elif self.mode == "symlink":
            real_operation_dir = operation_dir.with_name("real-operation")
            operation_dir.rename(real_operation_dir)
            operation_dir.symlink_to(real_operation_dir, target_is_directory=True)
        elif self.mode == "hash":
            manifest_ref["sha256"] = "0" * 64
        elif self.mode == "media":
            manifest_ref["media_type"] = "text/plain"
        elif self.mode == "size":
            manifest_ref["size"] += 1
        else:  # pragma: no cover - 测试配置错误
            raise AssertionError(f"unknown manifest attack: {self.mode}")
        return result


class _RecordingWriter:
    """模拟 Working Memory 的原子 BatchCreated 写入。"""

    def __init__(self, *, fail_once: bool = False, reader=None) -> None:
        self.calls = []
        self.fail_once = fail_once
        self.reader = reader

    def begin_next_investigation_batch(self, task_id, payload, source_refs):
        self.calls.append((task_id, payload, tuple(source_refs)))
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("Working Memory 暂时不可用")
        if self.reader is not None:
            self.reader.batches[payload["batch_id"]] = copy.deepcopy(payload)
            self.reader.batch_receipts[payload["batch_id"]] = {
                "event_id": "batch-created-event-1",
                "payload": copy.deepcopy(payload),
                "source_refs": list(source_refs),
            }
            self.reader.investigation_sessions[payload["batch_id"]] = [
                {
                    "combination_index": candidate["combination_index"],
                    "combination_id": candidate["combination_id"],
                    "workspace_ref": f"workspace/new/{candidate['combination_index']}",
                    "scope_id": f"scope-new-{candidate['combination_index']}",
                }
                for candidate in payload["candidates"]
                if candidate["selection_status"] == "selected"
            ]
        return "batch-created-event-1"


class _RecordingScopeCredentials:
    """记录逐 scope 撤销，并可模拟一次失败或不确定返回。"""

    def __init__(
        self,
        *,
        fail_calls: set[int] | None = None,
        uncertain_calls: set[int] | None = None,
    ) -> None:
        self.calls = []
        self.fail_calls = set(fail_calls or ())
        self.uncertain_calls = set(uncertain_calls or ())

    def revoke_investigation_scope_capability(self, **kwargs):
        """记录完整身份；指定序号返回异常或未注册结果。"""

        self.calls.append(dict(kwargs))
        call_index = len(self.calls)
        if call_index in self.fail_calls:
            raise RuntimeError(f"scope revoke failure {call_index}")
        if call_index in self.uncertain_calls:
            return {"uncertain": True}
        return None


def _bridge(
    tmp_path: Path,
    state,
    *,
    fail_writer: bool = False,
    scope_credentials=None,
    allow_test_scope_cleanup_bypass: bool = False,
):
    """持久化 state 并装配一份 bridge fixture。"""

    task_root = tmp_path / "hypothesis-task-root"
    reader = _configure_reader(_make_reader(), task_root=task_root)
    store = InvestigationSQLiteStore(tmp_path / "investigation.sqlite3")
    store.save_investigation_state(state)
    generator = _RecordingGenerator()
    writer = _RecordingWriter(fail_once=fail_writer, reader=reader)
    if scope_credentials is None and not allow_test_scope_cleanup_bypass:
        scope_credentials = _RecordingScopeCredentials()
    bridge = InvestigationHypothesisRegenerationBridge(
        investigation_store=store,
        working_memory_reader=reader,
        working_memory_writer=writer,
        hypothesis_generator=generator,
        scope_credentials=scope_credentials,
        allow_test_scope_cleanup_bypass=allow_test_scope_cleanup_bypass,
    )
    bridge._test_generation_config = replace(
        _generation_config(),
        task_root=str(task_root),
    )
    return bridge, store, generator, writer


def _run(
    bridge: InvestigationHypothesisRegenerationBridge,
    *,
    source_record_ids=(),
):
    """以固定显式身份执行一次重生。"""

    return bridge.regenerate_hypothesis_batch_from_investigation(
        task_id="task-1",
        next_batch_id="batch-2",
        next_batch_index=2,
        operation_instance_id="hypothesis-operation-2",
        session_identities=_identities(),
        generation_config=bridge._test_generation_config,
        source_record_ids=source_record_ids,
    )


def test_scope_cleanup_dependency_is_required_and_test_bypass_is_pending(
    tmp_path: Path,
) -> None:
    """生产缺失依赖立即失败，显式测试 bypass 也不能返回 completed。"""

    bridge, store, generator, writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
        allow_test_scope_cleanup_bypass=True,
    )
    with pytest.raises(ValueError, match="scope_credentials"):
        InvestigationHypothesisRegenerationBridge(
            investigation_store=store,
            working_memory_reader=bridge.working_memory_reader,
            working_memory_writer=writer,
            hypothesis_generator=generator,
        )

    outcome = _run(bridge)

    assert outcome.status == "pending"
    assert outcome.cleanup_status == "not_required"
    assert "测试 scope cleanup bypass" in (outcome.reason or "")
    receipt = store.load_hypothesis_regeneration_receipt(
        "task-1",
        "regenerate:task-1:batch-2",
    )
    assert receipt is not None
    assert receipt.receipt["cleanup"]["status"] == "not_required"


def test_bridge_constructor_rejects_unregistered_dependency_shapes(
    tmp_path: Path,
) -> None:
    """bridge 构造阶段拒绝错误依赖类型，避免运行中才出现外部副作用。"""

    bridge, _store, _generator, _writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
    )
    common = {
        "investigation_store": bridge.investigation_store,
        "working_memory_reader": bridge.working_memory_reader,
        "working_memory_writer": bridge.working_memory_writer,
        "hypothesis_generator": bridge.hypothesis_generator,
        "scope_credentials": bridge.scope_credentials,
    }
    for field in (
        "investigation_store",
        "working_memory_reader",
        "working_memory_writer",
        "hypothesis_generator",
    ):
        invalid = dict(common)
        invalid[field] = object()
        with pytest.raises(TypeError, match=field):
            InvestigationHypothesisRegenerationBridge(**invalid)
    with pytest.raises(TypeError, match="allow_test_scope_cleanup_bypass"):
        InvestigationHypothesisRegenerationBridge(
            **common,
            allow_test_scope_cleanup_bypass=1,
        )
    with pytest.raises(TypeError, match="scope_credentials"):
        InvestigationHypothesisRegenerationBridge(
            **{**common, "scope_credentials": object()},
        )


def test_not_required_receipt_migrates_when_scope_dependency_is_injected(
    tmp_path: Path,
) -> None:
    """历史测试 bypass receipt 在恢复时注入真实依赖会转为 cleanup_pending。"""

    bypass_bridge, store, generator, writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
        allow_test_scope_cleanup_bypass=True,
    )
    first = _run(bypass_bridge)
    assert first.status == "pending"

    scope_credentials = _RecordingScopeCredentials()
    migrated_bridge = InvestigationHypothesisRegenerationBridge(
        investigation_store=store,
        working_memory_reader=bypass_bridge.working_memory_reader,
        working_memory_writer=writer,
        hypothesis_generator=generator,
        scope_credentials=scope_credentials,
    )
    migrated_bridge._test_generation_config = bypass_bridge._test_generation_config
    second = _run(migrated_bridge)

    assert second.status == "completed"
    assert second.cleanup_status == "completed"
    assert len(scope_credentials.calls) == 5


def test_fewer_than_five_markers_are_rejected_before_generation(tmp_path: Path) -> None:
    """不足五个 marker 时不读取模型、不写 receipt，也不写 Working Memory。"""

    state = mark_all_refuted(make_initial_state(), "initial")
    sessions = list(state.sessions)
    sessions[0] = sessions[0].model_copy(
        update={"phase": "validated", "route_marker": None},
    )
    state = state.model_copy(update={"sessions": tuple(sessions), "route": "investigation"})
    bridge, store, generator, writer = _bridge(tmp_path, state)

    with pytest.raises(ValueError, match="五个.*marker"):
        _run(bridge)

    assert generator.calls == []
    assert writer.calls == []
    assert (
        store.load_hypothesis_regeneration_receipt("task-1", "regenerate:task-1:batch-2")
        is None
    )


def test_operation_key_cannot_be_supplied_by_calling_code(tmp_path: Path) -> None:
    """operation_key 必须由 task/batch 唯一派生，调用方不能注入另一个键。"""

    bridge, _store, generator, writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
    )
    with pytest.raises(ValueError, match="operation_key.*自定义"):
        bridge.regenerate_hypothesis_batch_from_investigation(
            task_id="task-1",
            next_batch_id="batch-2",
            next_batch_index=2,
            operation_instance_id="hypothesis-operation-2",
            session_identities=_identities(),
            generation_config=_generation_config(),
            operation_key="caller-chosen-key",
        )
    assert generator.calls == []
    assert writer.calls == []


def test_regeneration_rejects_invalid_next_batch_index_before_side_effects(
    tmp_path: Path,
) -> None:
    """下一批序号非法时不读取模型、不建立 receipt。"""

    bridge, store, generator, writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
    )
    with pytest.raises(ValueError, match="next_batch_index"):
        bridge.regenerate_hypothesis_batch_from_investigation(
            task_id="task-1",
            next_batch_id="batch-2",
            next_batch_index=1,
            operation_instance_id="hypothesis-operation-2",
            session_identities=_identities(),
            generation_config=bridge._test_generation_config,
        )
    assert generator.calls == []
    assert writer.calls == []
    assert store.load_hypothesis_regeneration_receipt(
        "task-1", "regenerate:task-1:batch-2"
    ) is None


def test_regeneration_rejects_secret_causal_source_before_side_effects(
    tmp_path: Path,
) -> None:
    """causal source 不能携带宿主路径或凭据文本。"""

    bridge, store, generator, writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
    )
    with pytest.raises(ValueError, match="causal_source_ref"):
        bridge.regenerate_hypothesis_batch_from_investigation(
            task_id="task-1",
            next_batch_id="batch-2",
            next_batch_index=2,
            operation_instance_id="hypothesis-operation-2",
            session_identities=_identities(),
            generation_config=bridge._test_generation_config,
            causal_source_ref="/tmp/provider-api-key=secret",
        )
    assert generator.calls == []
    assert writer.calls == []
    assert store.load_hypothesis_regeneration_receipt(
        "task-1", "regenerate:task-1:batch-2"
    ) is None


def test_regeneration_rejects_missing_investigation_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """没有已持久化 InvestigationState 时立即失败。"""

    bridge, _store, generator, writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
    )
    monkeypatch.setattr(
        bridge.investigation_store,
        "load_investigation_snapshot",
        lambda _task_id: None,
    )
    with pytest.raises(ValueError, match="尚未持久化"):
        _run(bridge)
    assert generator.calls == []
    assert writer.calls == []


def test_regeneration_rejects_state_task_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """持久状态的 task_id 漂移时不得进入模型或 receipt。"""

    bridge, store, generator, writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
    )
    snapshot = store.load_investigation_snapshot("task-1")
    assert snapshot is not None
    drifted = replace(
        snapshot,
        state=snapshot.state.model_copy(update={"task_id": "other-task"}),
    )
    monkeypatch.setattr(
        bridge.investigation_store,
        "load_investigation_snapshot",
        lambda _task_id: drifted,
    )
    with pytest.raises(ValueError, match="task_id"):
        _run(bridge)
    assert generator.calls == []
    assert writer.calls == []


def test_regeneration_rejects_saved_source_record_shape(
    tmp_path: Path,
) -> None:
    """已有 receipt 的 source_record_ids 必须保持数组形状。"""

    bridge, store, _generator, _writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
        fail_writer=True,
    )
    with pytest.raises(HypothesisRegenerationBridgeError):
        _run(bridge)
    saved = store.load_hypothesis_regeneration_receipt(
        "task-1", "regenerate:task-1:batch-2"
    )
    assert saved is not None
    tampered = copy.deepcopy(dict(saved.receipt))
    tampered["source_record_ids"] = "not-an-array"
    store.save_hypothesis_regeneration_receipt(
        "task-1",
        "regenerate:task-1:batch-2",
        tampered,
        expected_revision=saved.revision,
    )
    with pytest.raises(ValueError, match="source_record_ids"):
        _run(bridge)


@pytest.mark.parametrize("next_batch_index", [3, 4])
def test_regeneration_rejects_non_contiguous_working_memory_batch_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    next_batch_index: int,
) -> None:
    """Working Memory 批次序号或请求序号不连续时停止在模型前。"""

    bridge, _store, generator, writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
    )
    fake_memory = SimpleNamespace(
        closed_batches=(SimpleNamespace(batch_index=2),),
        current_batch=SimpleNamespace(batch_index=3),
        to_prompt_payload=lambda: {},
    )
    monkeypatch.setattr(
        bridge_module,
        "build_hypothesis_regeneration_working_memory_input",
        lambda *_args: fake_memory,
    )
    with pytest.raises(ValueError, match="批次序号不连续|当前批次不连续|next_batch_index"):
        bridge.regenerate_hypothesis_batch_from_investigation(
            task_id="task-1",
            next_batch_id="batch-2",
            next_batch_index=next_batch_index,
            operation_instance_id="hypothesis-operation-2",
            session_identities=_identities(),
            generation_config=bridge._test_generation_config,
        )
    assert generator.calls == []
    assert writer.calls == []


def test_regeneration_receipt_claim_competition_without_saved_receipt_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """receipt CAS 竞争后若读不到对手 receipt，不能猜测并继续。"""

    bridge, _store, generator, writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
    )
    monkeypatch.setattr(
        bridge,
        "_create_started_receipt",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(RuntimeError, match="竞争"):
        _run(bridge)
    assert generator.calls == []
    assert writer.calls == []


def test_regeneration_gate_marker_source_is_exactly_five_opaque_events() -> None:
    """route gate 只接受五个结构完整且不重复的 marker event。"""

    valid = {
        "allowed": True,
        "markers": [
            {"marker_event_id": f"marker-{index}"}
            for index in range(1, 6)
        ],
    }
    assert _read_regeneration_gate_marker_ids(valid) == [
        f"marker-{index}" for index in range(1, 6)
    ]
    with pytest.raises(ValueError, match="未满足"):
        _read_regeneration_gate_marker_ids({"allowed": False, "reason": "未满足"})
    with pytest.raises(ValueError, match="marker 来源"):
        _read_regeneration_gate_marker_ids({"allowed": True, "markers": "bad"})
    with pytest.raises(ValueError, match="结构非法"):
        _read_regeneration_gate_marker_ids(
            {"allowed": True, "markers": [None] * 5}
        )
    with pytest.raises(ValueError, match="五个"):
        _read_regeneration_gate_marker_ids(
            {"allowed": True, "markers": [{"marker_event_id": "same"}] * 5}
        )


def test_query_provenance_compatibility_and_grouped_helpers_cover_boundaries(
    tmp_path: Path,
) -> None:
    """旧单 Query 兼容路径和多 Query 分组路径都逐项核对身份覆盖。"""

    ranked = [
        {"hypothesis_id": "h-1"},
        {"hypothesis_id": "h-2"},
    ]
    assert _fallback_single_query_provenance(
        {"question_snapshot": {"initial_query_angles": ["q-1"]}},
        ranked,
        (),
        {"h-1"},
        "batch-1",
    ) == {"h-1": "q-1", "h-2": "q-1"}
    with pytest.raises(ValueError, match="selected hypothesis"):
        _fallback_single_query_provenance(
            {"question_snapshot": {"initial_query_angles": ["q-1"]}},
            ranked,
            (),
            {"missing"},
            "batch-1",
        )
    with pytest.raises(ValueError, match="source_query"):
        _fallback_single_query_provenance({}, ranked, (), set(), "batch-1")
    with pytest.raises(ValueError, match="必须是 object"):
        _fallback_single_query_provenance(
            {"question_snapshot": {"initial_query_angles": ["q-1"]}},
            [None],
            (),
            set(),
            "batch-1",
        )

    groups = [
        {"query": "q-1", "hypotheses": [{"hypothesis_id": "h-1"}]},
        {"query": "q-2", "hypotheses": [{"hypothesis_id": "h-2"}]},
    ]
    assert _grouped_query_provenance(
        groups,
        batch_id="batch-1",
        frozen_queries=("q-1", "q-2"),
        ranked_ids={"h-1", "h-2"},
        required_ids={"h-1"},
    ) == {"h-1": "q-1", "h-2": "q-2"}
    with pytest.raises(ValueError, match="重复 source_query"):
        _grouped_query_provenance(
            groups + [{"query": "q-1", "hypotheses": [{"hypothesis_id": "h-3"}]}],
            batch_id="batch-1",
            frozen_queries=("q-1", "q-2"),
            ranked_ids={"h-1", "h-2", "h-3"},
            required_ids=set(),
        )
    with pytest.raises(ValueError, match="完整 ranked"):
        _grouped_query_provenance(
            groups[:1],
            batch_id="batch-1",
            frozen_queries=("q-1",),
            ranked_ids={"h-1", "h-2"},
            required_ids=set(),
        )

    bridge, _store, _generator, _writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
    )
    reader = bridge.working_memory_reader
    ranked_value = copy.deepcopy(
        reader.batches["batch-1"]["hypothesis_phase"]["ranked_strategies"]
    )
    for index, item in enumerate(ranked_value, start=1):
        item["hypothesis_id"] = f"new-hypothesis-{index}"
    ranked_value[0].pop("rank", None)
    normalized = _normalize_ranked_hypotheses(
        {"ranked_strategies": ranked_value},
        generation_config=bridge._test_generation_config,
    )
    assert normalized[0]["rank"] == 1
    duplicate = copy.deepcopy(ranked_value)
    duplicate[1]["hypothesis_id"] = duplicate[0]["hypothesis_id"]
    with pytest.raises(ValueError, match="必须唯一"):
        _normalize_ranked_hypotheses(
            {"ranked_strategies": duplicate},
            generation_config=bridge._test_generation_config,
        )
    unallocated = copy.deepcopy(ranked_value)
    unallocated[0]["hypothesis_id"] = "not-allocated"
    with pytest.raises(ValueError, match="未分配"):
        _normalize_ranked_hypotheses(
            {"ranked_strategies": unallocated},
            generation_config=bridge._test_generation_config,
        )


def test_hypothesis_phase_coverage_accepts_consumed_merge_parents() -> None:
    """bridge 允许归并父项留在分组、仅由综合候选引用的正常结果。"""

    generation_config = HypothesisRegenerationGenerationConfig(
        task_root="local_data/runtime/bridge-test",
        stage_tool_projection=(),
        num_queries=1,
        num_hypotheses_per_query=5,
        hypothesis_ids_by_query=(
            tuple(f"h{index}" for index in range(1, 6)),
        ),
        merge_hypothesis_ids=("hm1", "hm2"),
    )

    def hypothesis(hypothesis_id: str, strategy_name: str) -> dict[str, str]:
        return {
            "hypothesis_id": hypothesis_id,
            "strategy_name": strategy_name,
            "reasoning": "固定计算单元下的科学判断。",
        }

    phase = {
        "queries": ["归并角度"],
        "hypotheses_by_query": [
            {
                "query": "归并角度",
                "hypotheses": [
                    hypothesis(f"h{index}", f"父项{index}")
                    for index in range(1, 6)
                ],
            }
        ],
        "ranked_strategies": [
            {
                **hypothesis("hm1", "综合候选一"),
                "synthesis_sources": [
                    {"parent_hypothesis_ids": ["h1", "h2"]}
                ],
            },
            {
                **hypothesis("hm2", "综合候选二"),
                "synthesis_sources": [
                    {"parent_hypothesis_ids": ["h1", "h2"]}
                ],
            },
            *[
                hypothesis(f"h{index}", f"父项{index}")
                for index in range(3, 6)
            ],
        ],
    }

    normalized = _normalize_hypothesis_phase(
        phase,
        generation_config=generation_config,
        expected_queries=("归并角度",),
    )

    assert [
        item["hypothesis_id"]
        for item in normalized["hypotheses_by_query"][0]["hypotheses"]
    ] == ["h1", "h2", "h3", "h4", "h5"]
    assert [
        item["hypothesis_id"] for item in normalized["ranked_strategies"]
    ] == ["hm1", "hm2", "h3", "h4", "h5"]

    orphan_parent = copy.deepcopy(phase)
    for item in orphan_parent["ranked_strategies"][:2]:
        item["synthesis_sources"][0]["parent_hypothesis_ids"] = ["h1"]
    with pytest.raises(ValueError, match="不在完整排名"):
        _normalize_hypothesis_phase(
            orphan_parent,
            generation_config=generation_config,
            expected_queries=("归并角度",),
        )


def test_legacy_combination_outcome_preserves_real_result_review_and_query(
    tmp_path: Path,
) -> None:
    """兼容 package 组合结果保留真实观察、限制、Review 理由和 source query。"""

    bridge, _store, _generator, _writer = _bridge(
        tmp_path,
        exhaust_three_repairs(make_initial_state()),
    )
    reader = bridge.working_memory_reader
    snapshot = bridge.investigation_store.load_investigation_snapshot("task-1")
    assert snapshot is not None
    working_memory = bridge_module.build_hypothesis_regeneration_working_memory_input(
        snapshot.state,
        reader,
    )
    batch = working_memory.current_batch
    combination = batch.combinations[0]
    findings, sessions, results = _index_legacy_batch_records(batch)
    source_query = {combination.hypothesis_id: "固定角度"}

    outcome = _compose_legacy_combination_outcome(
        batch,
        combination,
        query_by_hypothesis=source_query,
        findings=findings,
        sessions=sessions,
        results=results,
    )
    assert outcome["source_query"] == "固定角度"
    assert outcome["cited_evidence"]

    fallback_findings = copy.deepcopy(findings)
    fallback_findings[combination.combination_id]["repair_advice"] = ""
    fallback = _compose_legacy_combination_outcome(
        batch,
        combination,
        query_by_hypothesis=source_query,
        findings=fallback_findings,
        sessions=sessions,
        results=results,
    )
    assert fallback["final_judge"]["reason"]
    with pytest.raises(ValueError, match="source_query"):
        _compose_legacy_combination_outcome(
            batch,
            combination,
            query_by_hypothesis={},
            findings=findings,
            sessions=sessions,
            results=results,
        )
    missing_sessions = dict(sessions)
    missing_sessions.pop(combination.combination_id)
    with pytest.raises(ValueError, match="session"):
        _compose_legacy_combination_outcome(
            batch,
            combination,
            query_by_hypothesis=source_query,
            findings=findings,
            sessions=missing_sessions,
            results=results,
        )


def test_legacy_batch_outcomes_reject_incomplete_query_or_session_alignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """兼容 package 必须保留冻结 Query 覆盖和五个 session 对齐。"""

    bridge, _store, _generator, _writer = _bridge(
        tmp_path,
        exhaust_three_repairs(make_initial_state()),
    )
    snapshot = bridge.investigation_store.load_investigation_snapshot("task-1")
    assert snapshot is not None
    working_memory = bridge_module.build_hypothesis_regeneration_working_memory_input(
        snapshot.state,
        bridge.working_memory_reader,
    )
    batch = working_memory.current_batch
    selected_ids = tuple(item.hypothesis_id for item in batch.combinations)
    monkeypatch.setattr(
        bridge_module,
        "_query_by_hypothesis_id",
        lambda *_args, **_kwargs: {item: "固定角度" for item in selected_ids},
    )
    with pytest.raises(ValueError, match="不在首批冻结 Query"):
        _compose_legacy_batch_outcomes(
            batch,
            reader=bridge.working_memory_reader,
            state=snapshot.state,
            queries=("不是冻结 Query",),
        )
    with pytest.raises(ValueError, match="完整覆盖"):
        _compose_legacy_batch_outcomes(
            batch,
            reader=bridge.working_memory_reader,
            state=snapshot.state,
            queries=("固定角度", "缺少的 Query"),
        )
    removed_combination = batch.sessions[-1]["combination_id"]
    short_batch = batch.model_copy(
        update={
            "sessions": batch.sessions[:-1],
            "results": tuple(
                item
                for item in batch.results
                if item["combination_id"] != removed_combination
            ),
        }
    )
    with pytest.raises(ValueError, match="session/finding"):
        _compose_legacy_batch_outcomes(
            short_batch,
            reader=bridge.working_memory_reader,
            state=snapshot.state,
            queries=("固定角度",),
        )


def test_regeneration_input_keeps_hypotheses_results_and_every_review_round(
    tmp_path: Path,
) -> None:
    """下一批模型输入保留原始假设、真实结果引用和每一轮 DS Review。"""

    bridge, _store, _generator, _writer = _bridge(
        tmp_path,
        exhaust_three_repairs(make_initial_state()),
    )
    snapshot = bridge.investigation_store.load_investigation_snapshot("task-1")
    assert snapshot is not None
    working_memory = bridge_module.build_hypothesis_regeneration_working_memory_input(
        snapshot.state,
        bridge.working_memory_reader,
    )
    payload = working_memory.to_prompt_payload()
    assert payload["scientific_question"] == "冻结科学问题"
    assert payload["retrieval_context"]["literature_review"] == "冻结检索结论"
    batch = payload["current_batch"]
    assert len(batch["combinations"]) == 5
    assert batch["combinations"][0]["hypothesis"]["reasoning"]
    assert len(batch["results"]) == 20
    assert all(
        item["observations"] and item["artifact_refs"]
        for item in batch["results"]
    )
    assert len(batch["validations"]) == 4
    assert all(
        finding["repair_advice"]
        for validation in batch["validations"]
        for finding in validation["findings"]
    )
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "workspace_ref" not in serialized
    assert "codex_thread_id" not in serialized


def test_receipt_cleanup_validation_rejects_untrusted_progress_shapes(
    tmp_path: Path,
) -> None:
    """receipt 清理计划只接受 opaque source ids、可选文本和 object 计划。"""

    bridge, store, _generator, _writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
        allow_test_scope_cleanup_bypass=True,
    )
    assert _run(bridge).status == "pending"
    saved = store.load_hypothesis_regeneration_receipt(
        "task-1", "regenerate:task-1:batch-2"
    )
    assert saved is not None
    baseline = copy.deepcopy(dict(saved.receipt))
    cases = []
    malformed = copy.deepcopy(baseline)
    malformed["source_record_ids"] = "not-array"
    cases.append((malformed, "source_record_ids"))
    malformed = copy.deepcopy(baseline)
    malformed["source_record_ids"] = [1]
    cases.append((malformed, "source_record_ids"))
    malformed = copy.deepcopy(baseline)
    malformed["source_record_ids"] = ["opaque", "opaque"]
    cases.append((malformed, "opaque"))
    malformed = copy.deepcopy(baseline)
    malformed["causal_source_ref"] = 1
    cases.append((malformed, "causal_source_ref"))
    malformed = copy.deepcopy(baseline)
    malformed["cleanup"] = []
    cases.append((malformed, "cleanup"))
    for value, message in cases:
        with pytest.raises(ValueError, match=message):
            _validate_receipt_cleanup_fields(value)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"top_n": 4}, "严格选择五个"),
        ({"num_queries": 2}, "数量与 num_queries"),
        ({"hypothesis_ids_by_query": (("new-hypothesis-1",),)}, "每组数量"),
        ({"task_config_schema": "unknown"}, "task_config_schema"),
    ],
)
def test_generation_config_rejects_inconsistent_frozen_budget(
    updates: dict[str, object],
    message: str,
) -> None:
    """生成配置必须严格匹配五组合、Query 数量和 schema 注册表。"""

    with pytest.raises(ValueError, match=message):
        replace(_generation_config(), **updates)


def test_receipt_identity_validation_rejects_each_untrusted_shape(
    tmp_path: Path,
) -> None:
    """receipt 的身份、摘要和五个 claim 形状都必须逐项可验证。"""

    bridge, store, _generator, _writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
    )
    assert _run(bridge).status == "completed"
    saved = store.load_hypothesis_regeneration_receipt(
        "task-1", "regenerate:task-1:batch-2"
    )
    assert saved is not None
    cases = [
        ("next_batch_index", True, "next_batch_index"),
        ("input_digest", "not-a-digest", "input_digest"),
        ("operation_reference", [], "operation_reference"),
        ("session_identities", [], "session_identities"),
    ]
    for field, value, message in cases:
        tampered = copy.deepcopy(dict(saved.receipt))
        tampered[field] = value
        with pytest.raises(ValueError, match=message):
            bridge_module._validate_receipt_identity_fields(tampered)

    malformed_identity = copy.deepcopy(dict(saved.receipt))
    malformed_identity["session_identities"] = [{} for _ in range(5)]
    with pytest.raises(ValueError, match="session_identities"):
        _validate_receipt_identity_fields(malformed_identity)

    malformed_index = copy.deepcopy(dict(saved.receipt))
    malformed_index["session_identities"][0]["combination_index"] = 9
    with pytest.raises(ValueError, match="combination_index"):
        _validate_receipt_identity_fields(malformed_index)

    malformed_scope = copy.deepcopy(dict(saved.receipt))
    malformed_scope["session_identities"][0]["scope_id"] = ""
    with pytest.raises(ValueError, match="scope_id"):
        _validate_receipt_identity_fields(malformed_scope)


def test_task_root_reader_barrier_rejects_unsafe_frozen_metadata(
    tmp_path: Path,
) -> None:
    """模型 artifact 根目录只能使用 TaskCreated 的安全相对路径。"""

    bridge, _store, _generator, _writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
    )
    reader = bridge.working_memory_reader
    original_task_created = copy.deepcopy(reader.task_created)
    original_scratch_root = reader.scratch_root
    task_root = bridge._test_generation_config.task_root
    cases = [
        ("task_output_relative_dir", None, task_root, "relative_dir"),
        ("task_output_relative_dir", "../unsafe", task_root, "安全相对路径"),
        ("task_output_relative_dir", "https://example.invalid/root", task_root, "安全相对路径"),
        ("task_output_relative_dir", "bridge-test", str(tmp_path / "other"), "冻结输出根"),
    ]
    try:
        for field, value, actual_root, message in cases:
            reader.task_created = copy.deepcopy(original_task_created)
            if value is None:
                reader.task_created.pop(field, None)
            else:
                reader.task_created[field] = value
            with pytest.raises(ValueError, match=message):
                _validate_task_root_against_task_created(reader, "task-1", actual_root)
        reader.task_created = copy.deepcopy(original_task_created)
        reader.scratch_root = ""
        with pytest.raises(ValueError, match="scratch_root"):
            _validate_task_root_against_task_created(reader, "task-1", task_root)
    finally:
        reader.task_created = original_task_created
        reader.scratch_root = original_scratch_root


def test_initial_query_retrieval_reader_rejects_incomplete_snapshot(
    tmp_path: Path,
) -> None:
    """首批冻结 Query 和 Retrieval 缺失或变形时不得猜测输入。"""

    bridge, _store, _generator, _writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
    )
    reader = bridge.working_memory_reader
    state = make_initial_state()
    original = copy.deepcopy(reader.batches["batch-1"])
    cases = [
        ("question_snapshot", None, "question_snapshot"),
        ("initial_query_angles", [""], "initial_query_angles"),
        ("initial_query_angles", ["重复", "重复"], "重复"),
        ("initial_retrieval_phase", [], "initial_retrieval_phase"),
    ]
    try:
        for field, value, message in cases:
            reader.batches["batch-1"] = copy.deepcopy(original)
            if field == "question_snapshot":
                reader.batches["batch-1"].pop(field, None)
            else:
                reader.batches["batch-1"]["question_snapshot"][field] = value
            with pytest.raises(ValueError, match=message):
                _read_initial_query_and_retrieval(reader, state)
    finally:
        reader.batches["batch-1"] = original


def test_session_identity_reader_rejects_each_duplicate_dimension() -> None:
    """下一批五个 session 的位置和三类资源身份都必须唯一。"""

    identities = _identities()
    cases = [
        (identities[:4], "五个"),
        (tuple(replace(item, combination_index=1) if index == 1 else item for index, item in enumerate(identities)), "combination_index"),
        (tuple(replace(item, combination_id=identities[0].combination_id) if index == 1 else item for index, item in enumerate(identities)), "combination_id"),
        (tuple(replace(item, workspace_ref=identities[0].workspace_ref) if index == 1 else item for index, item in enumerate(identities)), "workspace_ref"),
        (tuple(replace(item, scope_id=identities[0].scope_id) if index == 1 else item for index, item in enumerate(identities)), "scope_id"),
    ]
    for value, message in cases:
        with pytest.raises(ValueError, match=message):
            _validate_session_identities(value)


def test_cleanup_plan_header_reader_rejects_each_unregistered_shape(
    tmp_path: Path,
) -> None:
    """scope cleanup receipt 头部必须固定绑定 source/target 和五项 scope。"""

    bridge, store, _generator, _writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
        allow_test_scope_cleanup_bypass=True,
    )
    assert _run(bridge).status == "pending"
    saved = store.load_hypothesis_regeneration_receipt(
        "task-1", "regenerate:task-1:batch-2"
    )
    assert saved is not None
    baseline = copy.deepcopy(saved.receipt["cleanup"])
    cases = []
    cases.append(([], "cleanup plan"))
    unknown = copy.deepcopy(baseline)
    unknown["unregistered"] = True
    cases.append((unknown, "未注册字段"))
    wrong_schema = copy.deepcopy(baseline)
    wrong_schema["schema"] = "other"
    cases.append((wrong_schema, "schema"))
    wrong_status = copy.deepcopy(baseline)
    wrong_status["status"] = "unknown"
    cases.append((wrong_status, "status"))
    wrong_source = copy.deepcopy(baseline)
    wrong_source["source_batch_id"] = "other"
    cases.append((wrong_source, "source batch"))
    wrong_target = copy.deepcopy(baseline)
    wrong_target["target_batch_id"] = "other"
    cases.append((wrong_target, "target batch"))
    wrong_scopes = copy.deepcopy(baseline)
    wrong_scopes["scopes"] = []
    cases.append((wrong_scopes, "五个 scope"))
    for value, message in cases:
        with pytest.raises(ValueError, match=message):
            _read_cleanup_plan_header(
                value,
                previous_batch_id="batch-1",
                next_batch_id="batch-2",
            )


def test_finding_reader_rejects_missing_or_duplicate_review_entries(
    tmp_path: Path,
) -> None:
    """批次最新 Review 必须提供恰好五个唯一 combination finding。"""

    bridge, _store, _generator, _writer = _bridge(
        tmp_path,
        exhaust_three_repairs(make_initial_state()),
    )
    snapshot = bridge.investigation_store.load_investigation_snapshot("task-1")
    assert snapshot is not None
    memory = bridge_module.build_hypothesis_regeneration_working_memory_input(
        snapshot.state,
        bridge.working_memory_reader,
    )
    payload = memory.current_batch.model_dump(mode="json")
    cases = []
    missing = copy.deepcopy(payload)
    missing.pop("validations", None)
    cases.append((missing, "BatchValidation"))
    malformed = copy.deepcopy(payload)
    malformed["validations"] = [{}]
    cases.append((malformed, "findings"))
    empty = copy.deepcopy(payload)
    empty["validations"][-1]["findings"] = []
    cases.append((empty, "五个"))
    duplicate = copy.deepcopy(payload)
    findings = duplicate["validations"][-1]["findings"]
    findings[1]["combination_id"] = findings[0]["combination_id"]
    cases.append((duplicate, "重复"))
    invalid = copy.deepcopy(payload)
    invalid["validations"][-1]["findings"][0] = []
    cases.append((invalid, "finding"))
    for value, message in cases:
        with pytest.raises(ValueError, match=message):
            _finding_by_combination(value)


def test_task_root_must_match_task_created_output_root(tmp_path: Path) -> None:
    """模型 operation 不能把 TaskCreated 输出根替换成任意路径。"""

    bridge, _store, generator, writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
    )

    with pytest.raises(ValueError, match="task_root|TaskCreated"):
        bridge.regenerate_hypothesis_batch_from_investigation(
            task_id="task-1",
            next_batch_id="batch-2",
            next_batch_index=2,
            operation_instance_id="hypothesis-operation-2",
            session_identities=_identities(),
            generation_config=replace(
                _generation_config(),
                task_root="../other-task",
            ),
        )

    assert generator.calls == []
    assert writer.calls == []


def test_same_operation_concurrent_claim_calls_generator_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一下一批并发只能有一个 receipt owner 进入模型。"""

    task_root = tmp_path / "hypothesis-task-root"
    reader = _configure_reader(_make_reader(), task_root=task_root)
    store = InvestigationSQLiteStore(tmp_path / "investigation.sqlite3")
    store.save_investigation_state(mark_all_refuted(make_initial_state(), "initial"))
    generator = _BlockingGenerator()
    writer = _RecordingWriter(reader=reader)
    original_claim = store.claim_hypothesis_regeneration_receipt
    claim_entered = threading.Event()
    release_claim = threading.Event()
    claim_count = 0
    claim_lock = threading.Lock()

    def race_started_claim(task_id, operation_key, receipt):
        nonlocal claim_count
        if receipt.get("status") == "started":
            with claim_lock:
                claim_count += 1
                if claim_count == 2:
                    claim_entered.set()
            if not release_claim.wait(timeout=5):
                raise RuntimeError("测试 receipt claim 等待超时")
        return original_claim(task_id, operation_key, receipt)

    monkeypatch.setattr(
        store,
        "claim_hypothesis_regeneration_receipt",
        race_started_claim,
    )
    bridge = InvestigationHypothesisRegenerationBridge(
        investigation_store=store,
        working_memory_reader=reader,
        working_memory_writer=writer,
        hypothesis_generator=generator,
        scope_credentials=_RecordingScopeCredentials(),
    )
    bridge._test_generation_config = replace(
        _generation_config(),
        task_root=str(task_root),
    )
    outcomes = []
    errors = []

    def run_once() -> None:
        try:
            outcomes.append(_run(bridge))
        except BaseException as exc:  # pragma: no cover - 仅用于汇总竞态结果
            errors.append(exc)

    first = threading.Thread(target=run_once)
    second = threading.Thread(target=run_once)
    first.start()
    second.start()
    assert claim_entered.wait(timeout=5)
    release_claim.set()
    assert generator.entered.wait(timeout=5)
    time.sleep(0.1)
    generator.release.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert len(generator.calls) == 1
    assert any(outcome.status == "completed" for outcome in outcomes)
    assert all(
        isinstance(error, HypothesisRegenerationBridgeError)
        or "receipt" in str(error)
        for error in errors
    )


def test_next_batch_cannot_reuse_historical_combination_identity(tmp_path: Path) -> None:
    """下一批不能只更换 workspace/scope 而复用旧 combination_id。"""

    bridge, store, generator, writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
    )
    identities = list(_identities())
    identities[0] = identities[0].__class__(
        combination_index=1,
        combination_id="combination-1",
        workspace_ref="workspace/new/reused-combination",
        scope_id="scope-new-reused-combination",
    )
    with pytest.raises(ValueError, match="combination_id.*复用历史"):
        bridge.regenerate_hypothesis_batch_from_investigation(
            task_id="task-1",
            next_batch_id="batch-2",
            next_batch_index=2,
            operation_instance_id="hypothesis-operation-2",
            session_identities=tuple(identities),
            generation_config=_generation_config(),
        )
    assert generator.calls == []
    assert writer.calls == []
    assert store.load_hypothesis_regeneration_receipt(
        "task-1", "regenerate:task-1:batch-2"
    ) is None


def test_regeneration_request_keeps_all_results_reviews_repairs_and_original_hypotheses(
    tmp_path: Path,
) -> None:
    """重生请求同时携带全部调查/Review/返修版本和 WM 原始假设正文。"""

    state = exhaust_three_repairs(make_initial_state())
    bridge, _store, generator, _writer = _bridge(tmp_path, state)

    outcome = _run(bridge)

    assert outcome.status == "completed"
    request = generator.calls[0]
    current = request.working_memory_payload["current_batch"]
    assert len(current["results"]) == 20
    assert len(current["validations"]) == 4
    assert current["combinations"][0]["hypothesis"]["reasoning"] == (
        "批次 batch-1 的原始结构化理由 1"
    )
    assert request.regeneration_package["prior_hypothesis_outcomes"]
    assert all(
        item["terminal_classification"] == "insufficient_evidence"
        for item in request.regeneration_package["prior_hypothesis_outcomes"]
    )


def test_three_batch_regeneration_provider_receives_cumulative_selected_experience(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """从第三批重生时，Provider 只收到三批已选组合的累计真实调查经验。"""

    state = exhaust_three_repairs(make_initial_state("batch-1"))
    state = begin_next_batch(state, "batch-2")
    state = mark_all_refuted(state, "batch-2")
    state = begin_next_batch(state, "batch-3")
    state = exhaust_three_repairs(state)

    bridge, _store, _generator, writer = _bridge(tmp_path, state)
    task_root = Path(bridge._test_generation_config.task_root)
    reader = _configure_reader(
        _make_reader(batch_ids=("batch-1", "batch-2", "batch-3")),
        task_root=task_root,
    )
    bridge.working_memory_reader = reader
    writer.reader = reader

    provider_calls: list[dict[str, object]] = []
    result_factory = _RecordingGenerator()
    agent = HypothesisAgent(comparison_workers=1)

    def run_hypothesis_operation(**kwargs):
        provider_calls.append(copy.deepcopy(kwargs))
        return result_factory._result(
            str(task_root),
            str(kwargs["operation_instance_id"]),
        )

    def replay_hypothesis_operation_from_finalized_manifest(**kwargs):
        return result_factory._result(
            str(task_root),
            str(kwargs["operation_instance_id"]),
        )

    monkeypatch.setattr(agent, "run_hypothesis_operation", run_hypothesis_operation)
    monkeypatch.setattr(
        agent,
        "replay_hypothesis_operation_from_finalized_manifest",
        replay_hypothesis_operation_from_finalized_manifest,
    )
    bridge.hypothesis_generator = HypothesisAgentRegenerationGenerator(agent)

    outcome = bridge.regenerate_hypothesis_batch_from_investigation(
        task_id="task-1",
        next_batch_id="batch-4",
        next_batch_index=4,
        operation_instance_id="hypothesis-operation-4",
        session_identities=_identities(),
        generation_config=bridge._test_generation_config,
    )

    assert outcome.status == "completed"
    assert len(provider_calls) == 1
    package = provider_calls[0]["regeneration_package"]
    assert isinstance(package, dict)
    working_memory = package["investigation_working_memory"]
    assert isinstance(working_memory, dict)
    assert [item["batch_id"] for item in working_memory["closed_batches"]] == [
        "batch-1",
        "batch-2",
    ]
    assert working_memory["current_batch"]["batch_id"] == "batch-3"

    batches = [*working_memory["closed_batches"], working_memory["current_batch"]]
    expected_versions = {
        "batch-1": [
            f"{prefix}-combination-{index}"
            for prefix in ("initial", "repair-1", "repair-2", "repair-3")
            for index in range(1, 6)
        ],
        "batch-2": [f"batch-2-combination-{index}" for index in range(1, 6)],
        "batch-3": [
            f"{prefix}-combination-{index}"
            for prefix in ("initial", "repair-1", "repair-2", "repair-3")
            for index in range(1, 6)
        ],
    }
    for batch in batches:
        batch_id = batch["batch_id"]
        selected_ids = {
            item["hypothesis_id"] for item in batch["combinations"]
        }
        assert selected_ids == {
            f"hyp-{batch_id}-{index}" for index in range(1, 6)
        }
        assert f"hyp-{batch_id}-6" not in json.dumps(batch, ensure_ascii=False)
        assert {
            item["result_version"] for item in batch["results"]
        } == set(expected_versions[batch_id])
        assert all(item["observations"] and item["artifact_refs"] for item in batch["results"])


def test_recovery_does_not_repeat_generation_or_batch_created(tmp_path: Path) -> None:
    """完成后的重放只读 receipt，不再次调用 generator 或 writer。"""

    bridge, _store, generator, writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
    )
    first = _run(bridge)
    second = _run(bridge)

    assert first.status == second.status == "completed"
    assert second.generation_called is False
    assert len(generator.calls) == 1
    assert len(writer.calls) == 1


def test_five_selected_candidates_align_with_explicit_next_sessions(tmp_path: Path) -> None:
    """BatchCreated 的五个 selected candidate 与 BeginNextBatch 会话逐值对齐。"""

    bridge, store, _generator, writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
    )
    outcome = _run(bridge)
    payload = writer.calls[0][1]
    selected = [
        candidate
        for candidate in payload["candidates"]
        if candidate["selection_status"] == "selected"
    ]

    assert len(selected) == 5
    assert [item["combination_id"] for item in selected] == [
        item.combination_id for item in _identities()
    ]
    assert [item["combination_index"] for item in selected] == [1, 2, 3, 4, 5]
    assert outcome.state is not None
    assert tuple(item.combination_id for item in outcome.state.sessions) == tuple(
        item.combination_id for item in mark_all_refuted(make_initial_state()).sessions
    )
    saved = store.load_investigation_snapshot("task-1")
    assert saved is not None
    assert saved.state.batch_id == "batch-1"
    assert writer.reader.investigation_sessions["batch-2"]


def test_next_batch_does_not_write_legacy_investigation_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """下一批只由 Working Memory writer 写入，不再写旧 InvestigationState。"""

    bridge, store, _generator, _writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
    )

    def reject_legacy_state_write(*_args, **_kwargs):
        raise AssertionError("bridge 不得再次写 InvestigationState")

    monkeypatch.setattr(store, "save_investigation_state", reject_legacy_state_write)

    outcome = _run(bridge)

    assert outcome.status == "completed"
    saved = store.load_investigation_snapshot("task-1")
    assert saved is not None
    assert saved.state.batch_id == "batch-1"


def test_writer_failure_leaves_generated_receipt_and_retry_skips_generator(tmp_path: Path) -> None:
    """WM 部分失败时保留生成结果，恢复只重试一次 BatchCreated 写入。"""

    bridge, store, generator, writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
        fail_writer=True,
    )
    with pytest.raises(HypothesisRegenerationBridgeError, match="Working Memory") as error:
        _run(bridge)
    assert error.value.outcome.status == "pending"
    receipt = store.load_hypothesis_regeneration_receipt("task-1", "regenerate:task-1:batch-2")
    assert receipt is not None
    assert receipt.receipt["status"] == "generated"

    recovered = _run(bridge)
    assert recovered.status == "completed"
    assert len(generator.calls) == 1
    assert len(writer.calls) == 2


def test_generated_receipt_payload_tamper_is_rejected_before_retry_writer(
    tmp_path: Path,
) -> None:
    """generated receipt 的 payload 摘要失配时不能先把篡改值写进 WM。"""

    bridge, store, _generator, writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
        fail_writer=True,
    )
    with pytest.raises(HypothesisRegenerationBridgeError, match="Working Memory"):
        _run(bridge)
    saved = store.load_hypothesis_regeneration_receipt(
        "task-1",
        "regenerate:task-1:batch-2",
    )
    assert saved is not None
    assert saved.receipt["status"] == "generated"

    tampered = copy.deepcopy(dict(saved.receipt))
    assert "batch_payload" not in tampered
    tampered["output_digest"] = "0" * 64
    store.save_hypothesis_regeneration_receipt(
        "task-1",
        "regenerate:task-1:batch-2",
        tampered,
        expected_revision=saved.revision,
    )

    with pytest.raises(ValueError, match="output|摘要|payload"):
        _run(bridge)
    assert len(writer.calls) == 1


def test_generated_receipt_cannot_predeclare_working_memory_event(
    tmp_path: Path,
) -> None:
    """generated receipt 只能保存 finalized manifest，不能伪造 WM event 游标。"""

    bridge, store, _generator, writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
        fail_writer=True,
    )
    with pytest.raises(HypothesisRegenerationBridgeError, match="Working Memory"):
        _run(bridge)
    saved = store.load_hypothesis_regeneration_receipt(
        "task-1", "regenerate:task-1:batch-2"
    )
    assert saved is not None and saved.receipt["status"] == "generated"
    tampered = copy.deepcopy(dict(saved.receipt))
    tampered["batch_event_id"] = "forged-before-writer"
    store.save_hypothesis_regeneration_receipt(
        "task-1",
        "regenerate:task-1:batch-2",
        tampered,
        expected_revision=saved.revision,
    )
    with pytest.raises(ValueError, match="batch_event_id"):
        _run(bridge)
    assert len(writer.calls) == 1


def test_completed_receipt_write_failure_recovers_after_investigation_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Investigation 已推进时重试只补 completed receipt，不重写 WM 或调用模型。"""

    bridge, store, generator, writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
    )
    original_save = store.save_hypothesis_regeneration_receipt
    failed = True

    def fail_completed_once(task_id, operation_key, receipt, expected_revision=0):
        nonlocal failed
        if failed and receipt.get("status") == "completed":
            failed = False
            raise RuntimeError("completed receipt 暂时不可用")
        return original_save(task_id, operation_key, receipt, expected_revision)

    monkeypatch.setattr(
        store,
        "save_hypothesis_regeneration_receipt",
        fail_completed_once,
    )
    with pytest.raises(HypothesisRegenerationBridgeError, match="completed|游标"):
        _run(bridge)

    saved = store.load_investigation_snapshot("task-1")
    assert saved is not None
    assert saved.state.batch_id == "batch-1"
    receipt = store.load_hypothesis_regeneration_receipt(
        "task-1",
        "regenerate:task-1:batch-2",
    )
    assert receipt is not None
    assert receipt.receipt["status"] == "wm_written"

    recovered = _run(bridge)

    assert recovered.status == "completed"
    assert recovered.generation_called is False
    assert len(generator.calls) == 1
    assert len(writer.calls) == 1
    receipt = store.load_hypothesis_regeneration_receipt(
        "task-1",
        "regenerate:task-1:batch-2",
    )
    assert receipt is not None
    assert receipt.receipt["status"] == "completed"


def test_completed_receipt_revision_conflict_reuses_completed_competitor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """completed 游标 CAS 冲突时复用已经完成的对手 receipt。"""

    bridge, store, _generator, _writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
    )
    assert _run(bridge).status == "completed"
    receipt = store.load_hypothesis_regeneration_receipt(
        "task-1", "regenerate:task-1:batch-2"
    )
    snapshot = store.load_investigation_snapshot("task-1")
    assert receipt is not None
    assert snapshot is not None

    def raise_conflict(*_args, **_kwargs):
        raise InvestigationRevisionConflict("task-1", receipt.revision, receipt.revision + 1)

    monkeypatch.setattr(store, "save_hypothesis_regeneration_receipt", raise_conflict)
    reused = bridge._save_completed_receipt(
        receipt,
        batch_event_id=str(receipt.receipt["batch_event_id"]),
        state=snapshot.state,
    )

    assert reused.receipt["status"] == "completed"
    assert reused.revision == receipt.revision


def test_completed_receipt_restores_previous_history_view_without_writing_state() -> None:
    """已推进状态恢复 receipt 时只构造前批只读视图，历史顺序不符即失败。"""

    state = make_state_with_closed_history()
    restored = (
        InvestigationHypothesisRegenerationBridge._restore_previous_state_for_completed_receipt(
            state,
            previous_batch_id="batch-1",
        )
    )
    assert restored.batch_id == "batch-1"
    assert restored.history == ()
    with pytest.raises(ValueError, match="最后一个历史批次"):
        InvestigationHypothesisRegenerationBridge._restore_previous_state_for_completed_receipt(
            state,
            previous_batch_id="missing-batch",
        )


def test_wm_commit_cursor_persists_only_event_identity_after_begin_next_batch(
    tmp_path: Path,
) -> None:
    """WM 成功后 receipt 只追加 wm_written event 游标，不保存 BatchCreated 正文。"""

    bridge, store, _generator, _writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
        fail_writer=True,
    )
    with pytest.raises(HypothesisRegenerationBridgeError, match="Working Memory"):
        _run(bridge)
    receipt = store.load_hypothesis_regeneration_receipt(
        "task-1", "regenerate:task-1:batch-2"
    )
    snapshot = store.load_investigation_snapshot("task-1")
    assert receipt is not None and snapshot is not None
    cursor = bridge._save_wm_commit_cursor(
        receipt,
        batch_event_id="batch-created-event-manual",
        state=snapshot.state,
        generation_called=False,
    )
    assert cursor.receipt["status"] == "wm_written"
    assert cursor.receipt["batch_event_id"] == "batch-created-event-manual"
    assert "batch_payload" not in cursor.receipt


def test_wm_commit_cursor_reuses_competing_receipt_after_revision_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """wm_written CAS 冲突时复用已保存 receipt，不伪造第二个游标。"""

    bridge, store, _generator, _writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
        fail_writer=True,
    )
    with pytest.raises(HypothesisRegenerationBridgeError, match="Working Memory"):
        _run(bridge)
    receipt = store.load_hypothesis_regeneration_receipt(
        "task-1", "regenerate:task-1:batch-2"
    )
    snapshot = store.load_investigation_snapshot("task-1")
    assert receipt is not None and snapshot is not None

    def raise_conflict(*_args, **_kwargs):
        raise InvestigationRevisionConflict("task-1", receipt.revision, receipt.revision + 1)

    monkeypatch.setattr(
        store,
        "save_hypothesis_regeneration_receipt",
        raise_conflict,
    )
    competing = bridge._save_wm_commit_cursor(
        receipt,
        batch_event_id="batch-created-event-competing",
        state=snapshot.state,
        generation_called=False,
    )
    assert competing.receipt["status"] == "generated"


def test_scope_cleanup_waits_for_working_memory_commit(tmp_path: Path) -> None:
    """WM 尚未确认时不撤销旧 scope，恢复后才逐项完成清理。"""

    scope_credentials = _RecordingScopeCredentials()
    bridge, store, _generator, _writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
        fail_writer=True,
        scope_credentials=scope_credentials,
    )

    with pytest.raises(HypothesisRegenerationBridgeError, match="Working Memory"):
        _run(bridge)
    assert scope_credentials.calls == []
    receipt = store.load_hypothesis_regeneration_receipt(
        "task-1",
        "regenerate:task-1:batch-2",
    )
    assert receipt is not None
    assert receipt.receipt["status"] == "generated"
    assert receipt.receipt["cleanup"]["status"] == "cleanup_pending"
    assert all(
        item["status"] == "pending"
        for item in receipt.receipt["cleanup"]["scopes"]
    )

    outcome = _run(bridge)

    assert outcome.status == "completed"
    assert outcome.cleanup_status == "completed"
    assert len(scope_credentials.calls) == 5
    assert [call["batch_id"] for call in scope_credentials.calls] == [
        "batch-1"
    ] * 5
    assert [call["scope_id"] for call in scope_credentials.calls] == [
        f"scope-batch-1-combination-{index}" for index in range(1, 6)
    ]
    repeated = _run(bridge)
    assert repeated.status == "completed"
    assert repeated.cleanup_status == "completed"
    assert len(scope_credentials.calls) == 5


def test_scope_cleanup_waits_for_begin_next_batch_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BeginNextBatch 保存失败时不撤销旧 scope，恢复只推进尚未确认阶段。"""

    scope_credentials = _RecordingScopeCredentials()
    bridge, store, _generator, _writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
        scope_credentials=scope_credentials,
    )
    original_begin = bridge.working_memory_writer.begin_next_investigation_batch
    failed = True

    def fail_next_batch_once(task_id, payload, source_refs):
        nonlocal failed
        if failed:
            failed = False
            raise RuntimeError("begin_next_investigation_batch 暂时不可用")
        return original_begin(task_id, payload, source_refs)

    monkeypatch.setattr(
        bridge.working_memory_writer,
        "begin_next_investigation_batch",
        fail_next_batch_once,
    )
    with pytest.raises(HypothesisRegenerationBridgeError, match="Working Memory"):
        _run(bridge)
    assert scope_credentials.calls == []
    receipt = store.load_hypothesis_regeneration_receipt(
        "task-1",
        "regenerate:task-1:batch-2",
    )
    assert receipt is not None
    assert receipt.receipt["status"] == "generated"

    monkeypatch.setattr(
        bridge.working_memory_writer,
        "begin_next_investigation_batch",
        original_begin,
    )
    outcome = _run(bridge)

    assert outcome.status == "completed"
    assert len(scope_credentials.calls) == 5


def test_scope_cleanup_retries_only_unfinished_items_after_failure(tmp_path: Path) -> None:
    """单个 scope 撤销失败后，恢复不重复已保存为 revoked 的项目。"""

    scope_credentials = _RecordingScopeCredentials(fail_calls={2})
    bridge, store, _generator, _writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
        scope_credentials=scope_credentials,
    )

    first = _run(bridge)

    assert first.status == "pending"
    assert first.cleanup_status == "cleanup_pending"
    assert [status for _combination, status in first.cleanup_scope_statuses] == [
        "revoked",
        "pending",
        "pending",
        "pending",
        "pending",
    ]
    assert len(scope_credentials.calls) == 2

    second = _run(bridge)

    assert second.status == "completed"
    assert second.cleanup_status == "completed"
    assert len(scope_credentials.calls) == 6
    assert [call["combination_id"] for call in scope_credentials.calls] == [
        "combination-1",
        "combination-2",
        "combination-2",
        "combination-3",
        "combination-4",
        "combination-5",
    ]
    receipt = store.load_hypothesis_regeneration_receipt(
        "task-1",
        "regenerate:task-1:batch-2",
    )
    assert receipt is not None
    assert receipt.receipt["cleanup"]["status"] == "completed"
    assert all(
        item["status"] == "revoked"
        for item in receipt.receipt["cleanup"]["scopes"]
    )


def test_scope_cleanup_unknown_revoke_result_fails_closed(tmp_path: Path) -> None:
    """撤销服务返回未知值时保留 pending，下一次恢复才可确认全部清理。"""

    scope_credentials = _RecordingScopeCredentials(uncertain_calls={1})
    bridge, _store, _generator, _writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
        scope_credentials=scope_credentials,
    )

    first = _run(bridge)

    assert first.status == "pending"
    assert first.cleanup_status == "cleanup_pending"
    assert first.cleanup_scope_statuses[0] == ("combination-1", "pending")

    scope_credentials.uncertain_calls.clear()
    second = _run(bridge)

    assert second.status == "completed"
    assert second.cleanup_status == "completed"
    assert len(scope_credentials.calls) == 6


def test_started_receipt_replays_completed_operation_without_generating_again(tmp_path: Path) -> None:
    """模型已完成但 generated receipt 尚未写出时，只走同一 operation 的严格恢复接口。"""

    state = mark_all_refuted(make_initial_state(), "initial")
    bridge, store, generator, writer = _bridge(tmp_path, state)
    snapshot = store.load_investigation_snapshot("task-1")
    assert snapshot is not None
    request = bridge._build_generation_request_from_state(
        snapshot.state,
        task_id="task-1",
        next_batch_id="batch-2",
        next_batch_index=2,
        operation_key="regenerate:task-1:batch-2",
        operation_instance_id="hypothesis-operation-2",
        generation_config=bridge._test_generation_config,
    )
    started = bridge._create_started_receipt(
        snapshot,
        request=request,
        identities=_identities(),
        source_record_ids=(),
        causal_source_ref=None,
    )
    assert started is not None
    generator.recovery_result = generator._result(
        bridge._test_generation_config.task_root,
        "hypothesis-operation-2",
    )

    outcome = _run(bridge)

    assert outcome.status == "completed"
    assert generator.calls == []
    assert len(generator.recovery_calls) == 1
    assert len(writer.calls) == 1
    receipt = store.load_hypothesis_regeneration_receipt(
        "task-1",
        "regenerate:task-1:batch-2",
    )
    assert receipt is not None
    assert receipt.receipt["status"] == "completed"


def test_started_receipt_busy_recovery_stays_pending(tmp_path: Path) -> None:
    """活动 operation 的并发恢复保持 pending，不冻结 started receipt 为 failed。"""

    state = mark_all_refuted(make_initial_state(), "initial")
    bridge, store, generator, _writer = _bridge(tmp_path, state)
    snapshot = store.load_investigation_snapshot("task-1")
    assert snapshot is not None
    request = bridge._build_generation_request_from_state(
        snapshot.state,
        task_id="task-1",
        next_batch_id="batch-2",
        next_batch_index=2,
        operation_key="regenerate:task-1:batch-2",
        operation_instance_id="hypothesis-operation-2",
        generation_config=bridge._test_generation_config,
    )
    started = bridge._create_started_receipt(
        snapshot,
        request=request,
        identities=_identities(),
        source_record_ids=(),
        causal_source_ref=None,
    )
    assert started is not None

    def raise_busy(_request, _receipt):
        raise RuntimeError("operation still in progress")

    generator.recover_hypothesis_regeneration = raise_busy

    outcome = _run(bridge)

    assert outcome.status == "pending"
    assert outcome.receipt_status == "started"
    receipt = store.load_hypothesis_regeneration_receipt(
        "task-1",
        "regenerate:task-1:batch-2",
    )
    assert receipt is not None
    assert receipt.receipt["status"] == "started"


def test_initial_busy_generation_keeps_started_receipt_pending(
    tmp_path: Path,
) -> None:
    """初始生成遇到同一 operation busy 时不能冻结为 failed。"""

    bridge, store, generator, writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
    )

    def raise_busy(_request):
        generator.calls.append(_request)
        raise RuntimeError("operation busy")

    generator.generate_hypothesis_regeneration = raise_busy

    with pytest.raises(HypothesisRegenerationBridgeError) as error:
        _run(bridge)

    assert error.value.outcome.status == "pending"
    assert error.value.outcome.receipt_status == "started"
    assert writer.calls == []
    receipt = store.load_hypothesis_regeneration_receipt(
        "task-1",
        "regenerate:task-1:batch-2",
    )
    assert receipt is not None
    assert receipt.receipt["status"] == "started"

    recovered = _run(bridge)

    assert recovered.status == "completed"
    assert len(generator.calls) == 1
    assert len(writer.calls) == 1


def test_receipt_persists_and_verifies_complete_operation_reference(tmp_path: Path) -> None:
    """恢复必须核对完整 operation reference，而不是只看 operation_instance_id。"""

    bridge, store, generator, writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
    )
    first = _run(bridge)
    assert first.status == "completed"
    saved = store.load_hypothesis_regeneration_receipt(
        "task-1", "regenerate:task-1:batch-2"
    )
    assert saved is not None
    assert saved.receipt["operation_reference"]["operation"] == "hypothesis_generation"
    tampered = copy.deepcopy(dict(saved.receipt))
    tampered["operation_reference"]["task_root"] = "different-task-root"
    store.save_hypothesis_regeneration_receipt(
        "task-1",
        "regenerate:task-1:batch-2",
        tampered,
        expected_revision=saved.revision,
    )
    with pytest.raises(ValueError, match="operation reference"):
        _run(bridge)
    assert len(generator.calls) == 1
    assert len(writer.calls) == 1


def test_receipt_rejects_canonical_input_digest_tamper(tmp_path: Path) -> None:
    """receipt 的完整 canonical input 摘要被篡改时恢复必须停止。"""

    bridge, store, _generator, _writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
    )
    assert _run(bridge).status == "completed"
    saved = store.load_hypothesis_regeneration_receipt(
        "task-1",
        "regenerate:task-1:batch-2",
    )
    assert saved is not None
    tampered = copy.deepcopy(dict(saved.receipt))
    tampered["input_digest"] = "0" * 64
    store.save_hypothesis_regeneration_receipt(
        "task-1",
        "regenerate:task-1:batch-2",
        tampered,
        expected_revision=saved.revision,
    )

    with pytest.raises(ValueError, match="input digest|canonical input|摘要"):
        _run(bridge)


def test_receipt_rejects_batch_payload_tamper(tmp_path: Path) -> None:
    """receipt 不保存第二份 BatchCreated payload，篡改 manifest 摘要时停止。"""

    bridge, store, _generator, _writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
    )
    assert _run(bridge).status == "completed"
    saved = store.load_hypothesis_regeneration_receipt(
        "task-1",
        "regenerate:task-1:batch-2",
    )
    assert saved is not None
    tampered = copy.deepcopy(dict(saved.receipt))
    assert "batch_payload" not in tampered
    tampered["output_digest"] = "0" * 64
    store.save_hypothesis_regeneration_receipt(
        "task-1",
        "regenerate:task-1:batch-2",
        tampered,
        expected_revision=saved.revision,
    )

    with pytest.raises(ValueError, match="payload|output|BatchCreated|manifest|摘要"):
        _run(bridge)


def test_receipt_rejects_source_ref_or_event_tamper(tmp_path: Path) -> None:
    """恢复必须逐值核对 Working Memory 保存的 source_refs 与 event_id。"""

    bridge, store, _generator, _writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
    )
    assert _run(bridge).status == "completed"
    saved = store.load_hypothesis_regeneration_receipt(
        "task-1",
        "regenerate:task-1:batch-2",
    )
    assert saved is not None
    tampered = copy.deepcopy(dict(saved.receipt))
    tampered["source_refs"] = ["other-operation/manifest.json"]
    tampered["output_digest"] = _manifest_source_digest(tampered["source_refs"])
    tampered["batch_event_id"] = "other-event"
    store.save_hypothesis_regeneration_receipt(
        "task-1",
        "regenerate:task-1:batch-2",
        tampered,
        expected_revision=saved.revision,
    )

    with pytest.raises(ValueError, match="event_id|source_refs|事实"):
        _run(bridge)


def test_generated_receipt_source_ref_tamper_cannot_redirect_wm_write(
    tmp_path: Path,
) -> None:
    """模型结果已冻结但 WM 尚未写入时，篡改 source ref 也不能导入其它 operation。"""

    bridge, store, _generator, writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
        fail_writer=True,
    )
    with pytest.raises(HypothesisRegenerationBridgeError, match="Working Memory"):
        _run(bridge)
    saved = store.load_hypothesis_regeneration_receipt(
        "task-1",
        "regenerate:task-1:batch-2",
    )
    assert saved is not None and saved.receipt["status"] == "generated"
    tampered = copy.deepcopy(dict(saved.receipt))
    tampered["source_refs"] = [
        "working_memory/hypothesis_generation_inputs/other-operation/"
        "manifest_" + "0" * 64 + ".json"
    ]
    tampered["output_digest"] = _manifest_source_digest(tampered["source_refs"])
    store.save_hypothesis_regeneration_receipt(
        "task-1",
        "regenerate:task-1:batch-2",
        tampered,
        expected_revision=saved.revision,
    )

    with pytest.raises(ValueError, match="source_refs|operation"):
        _run(bridge)
    assert len(writer.calls) == 1


def test_completed_receipt_revision_tamper_fails_closed(tmp_path: Path) -> None:
    """completed receipt 不接受旧 Investigation revision 字段。"""

    bridge, store, generator, writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
    )
    assert _run(bridge).status == "completed"
    saved = store.load_hypothesis_regeneration_receipt(
        "task-1",
        "regenerate:task-1:batch-2",
    )
    assert saved is not None
    tampered = copy.deepcopy(dict(saved.receipt))
    tampered["investigation_revision"] = 999
    store.save_hypothesis_regeneration_receipt(
        "task-1",
        "regenerate:task-1:batch-2",
        tampered,
        expected_revision=saved.revision,
    )

    with pytest.raises(ValueError, match="字段|schema"):
        _run(bridge)
    assert len(generator.calls) == 1
    assert len(writer.calls) == 1


@pytest.mark.parametrize(
    "source_record_id",
    [
        "/tmp/source",
        "../source",
        "source/record",
        r"C:\\tmp\\source",
        "file://source-record",
        "working_memory/source",
        ".",
        "..",
        "record.with.dot",
        "record:with:colon",
    ],
)
def test_source_record_ids_must_be_opaque_identities(
    tmp_path: Path,
    source_record_id: str,
) -> None:
    """generation_source 只接受不含路径语义的 opaque identity。"""

    bridge, store, generator, writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
    )

    with pytest.raises(ValueError, match="source_record_id"):
        _run(bridge, source_record_ids=(source_record_id,))

    assert store.load_hypothesis_regeneration_receipt(
        "task-1",
        "regenerate:task-1:batch-2",
    ) is None
    assert generator.calls == []
    assert writer.calls == []


@pytest.mark.parametrize("attack", ["traversal", "cross_operation", "symlink", "hash", "media", "size"])
def test_manifest_ref_is_bound_to_task_operation_and_content(
    tmp_path: Path,
    attack: str,
) -> None:
    """manifest 引用必须绑定冻结 task root/operation，并逐项核对文件内容。"""

    bridge, _store, _normal_generator, writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
    )
    attack_generator = _ManifestAttackGenerator(attack)
    bridge.hypothesis_generator = attack_generator

    with pytest.raises(HypothesisRegenerationBridgeError, match="生成失败"):
        _run(bridge)
    assert len(attack_generator.calls) == 1
    assert writer.calls == []


def test_source_query_groups_are_checked_against_frozen_queries_and_selected_ids(
    tmp_path: Path,
) -> None:
    """source_query 重复、未知或 selected hypothesis 缺失时不能启动模型。"""

    bridge, _store, generator, writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
    )
    reader = bridge.working_memory_reader
    ranked = copy.deepcopy(reader.batches["batch-1"]["hypothesis_phase"]["ranked_strategies"])
    reader.batches["batch-1"]["hypothesis_phase"]["hypotheses_by_query"] = [
        {"query": "未知角度", "hypotheses": ranked}
    ]
    with pytest.raises(ValueError, match="未知 source query"):
        _run(bridge)
    assert generator.calls == []
    assert writer.calls == []


def test_source_query_declaration_without_groups_fails_closed(tmp_path: Path) -> None:
    """历史只声明 queries 却缺少分组时不能用单 Query 兼容猜测来源。"""

    bridge, _store, generator, writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
    )
    reader = bridge.working_memory_reader
    reader.batches["batch-1"]["hypothesis_phase"]["queries"] = ["固定角度"]
    with pytest.raises(ValueError, match="缺少 hypotheses_by_query source_query"):
        _run(bridge)
    assert generator.calls == []
    assert writer.calls == []


def test_source_query_groups_must_cover_every_ranked_hypothesis(tmp_path: Path) -> None:
    """source_query provenance 必须覆盖完整排名，不能只覆盖五个 selected。"""

    bridge, _store, generator, writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
    )
    reader = bridge.working_memory_reader
    ranked = copy.deepcopy(reader.batches["batch-1"]["hypothesis_phase"]["ranked_strategies"])
    reader.batches["batch-1"]["hypothesis_phase"]["hypotheses_by_query"] = [
        {"query": "固定角度", "hypotheses": ranked[:-1]}
    ]
    with pytest.raises(ValueError, match="完整 ranked hypothesis provenance"):
        _run(bridge)
    assert generator.calls == []
    assert writer.calls == []


def test_hypothesis_agent_rejects_private_investigation_memory_fields(tmp_path: Path) -> None:
    """HypothesisAgent 公共边界拒绝 thread 等私有字段和批次漂移。"""

    bridge, _store, generator, _writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
    )
    snapshot = bridge.investigation_store.load_investigation_snapshot("task-1")
    assert snapshot is not None
    request = bridge._build_generation_request_from_state(
        snapshot.state,
        task_id="task-1",
        next_batch_id="batch-2",
        next_batch_index=2,
        operation_key="regenerate:task-1:batch-2",
        operation_instance_id="hypothesis-operation-2",
        generation_config=_generation_config(),
    )
    package = copy.deepcopy(request.regeneration_package)
    package["investigation_working_memory"] = copy.deepcopy(
        request.working_memory_payload
    )
    package["investigation_working_memory"]["current_batch"]["sessions"][0][
        "thread"
    ] = "private-thread"
    with pytest.raises(ValueError, match="私有运行字段|私有会话身份"):
        HypothesisAgent._validate_regeneration_package(
            package,
            memory_enabled=True,
            expected_task_id="task-1",
            expected_batch_id="batch-1",
        )

    package = copy.deepcopy(request.regeneration_package)
    package["investigation_working_memory"] = copy.deepcopy(
        request.working_memory_payload
    )
    package["investigation_working_memory"]["batch_id"] = "batch-drift"
    with pytest.raises(ValueError, match="batch_id"):
        HypothesisAgent._validate_regeneration_package(
            package,
            memory_enabled=True,
            expected_task_id="task-1",
            expected_batch_id="batch-1",
        )


@pytest.mark.parametrize(
    ("location", "value", "message"),
    [
        (
            "scientific_context.chemistry_context",
            {"task_id": "task-leak"},
            "内部字段|未注册",
        ),
        (
            "scientific_context.initial_retrieval_phase",
            {"manifest_path": "/private/manifest.json"},
            "内部字段|宿主路径|未注册",
        ),
        (
            "scientific_context.chemistry_context",
            {"notes": "api_key=should-never-cross-agent"},
            "凭据|未注册",
        ),
    ],
)
def test_hypothesis_agent_scans_entire_public_package(
    tmp_path: Path,
    location: str,
    value: dict[str, str],
    message: str,
) -> None:
    """旧兼容 package 的嵌套上下文同样不能携带 schema、路径或 secret。"""

    bridge, _store, _generator, _writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
    )
    snapshot = bridge.investigation_store.load_investigation_snapshot("task-1")
    assert snapshot is not None
    request = bridge._build_generation_request_from_state(
        snapshot.state,
        task_id="task-1",
        next_batch_id="batch-2",
        next_batch_index=2,
        operation_key="regenerate:task-1:batch-2",
        operation_instance_id="hypothesis-operation-2",
        generation_config=bridge._test_generation_config,
    )
    package = copy.deepcopy(request.regeneration_package)
    target = package
    path_parts = location.split(".")
    for part in path_parts:
        target = target[part]
    target.update(value)
    with pytest.raises(ValueError, match=message):
        HypothesisAgent._validate_regeneration_package(
            package,
            memory_enabled=True,
            expected_task_id="task-1",
            expected_batch_id="batch-1",
        )


def test_finalized_hypothesis_manifest_replays_without_provider(tmp_path: Path) -> None:
    """真实 HypothesisAgent replay 只消费 finalized manifest，不再次调用 provider。"""

    arguments = _hypothesis_arguments()
    arguments["initial_query_angles"] = [
        _long_query("conformation spectroscopy"),
        _long_query("carbonyl vibration assignment"),
    ]
    arguments["regeneration_package"] = _regeneration_package(
        arguments,
        memory_enabled=True,
    )
    arguments["memory_enabled"] = True
    calls = []
    agent = HypothesisAgent(
        comparison_workers=1,
        model_provider=_catalog_provider(calls),
    )
    retrieval = arguments.pop("initial_retrieval_phase")
    first = agent.run_hypothesis_operation(
        task_root=str(tmp_path),
        operation_instance_id="hypothesis-replay-operation",
        initial_retrieval_phase=retrieval,
        expected_task_id="task-replay",
        expected_batch_id="batch-1",
        **arguments,
    )
    package = copy.deepcopy(arguments["regeneration_package"])
    base_input = copy.deepcopy(package)
    base_input.pop("source_record_ids", None)
    base_input["stage_tool_projection"] = copy.deepcopy(
        arguments["stage_tool_projection"]
    )
    base_input["chemical_capability_projection"] = copy.deepcopy(
        arguments["chemical_capability_projection"]
    )
    provider_call_count = len(calls)

    replayed = agent.replay_hypothesis_operation_from_finalized_manifest(
        task_root=str(tmp_path),
        operation_instance_id="hypothesis-replay-operation",
        base_input=base_input,
        scientific_question=arguments["scientific_question"],
        initial_query_angles=arguments["initial_query_angles"],
        initial_retrieval_phase=first["retrieval_result"],
        chemistry_context=arguments["chemistry_context"],
        num_queries=arguments["num_queries"],
        num_hypotheses_per_query=arguments["num_hypotheses_per_query"],
        top_n=arguments["top_n"],
        hypothesis_ids_by_query=arguments["hypothesis_ids_by_query"],
        merge_hypothesis_ids=arguments["merge_hypothesis_ids"],
        regeneration_package=package,
        memory_enabled=True,
        expected_task_id="task-replay",
        expected_batch_id="batch-1",
    )

    assert replayed is not None
    assert replayed["hypothesis_result"] == first["hypothesis_result"]
    assert len(calls) == provider_call_count


def test_finalized_hypothesis_manifest_rejects_capability_drift_without_provider(
    tmp_path: Path,
) -> None:
    """恢复时原子能力事实变化必须在读取 finalized 结果前失败关闭。"""

    arguments = _hypothesis_arguments()
    arguments["initial_query_angles"] = [
        _long_query("conformation spectroscopy"),
        _long_query("carbonyl vibration assignment"),
    ]
    arguments["regeneration_package"] = _regeneration_package(
        arguments,
        memory_enabled=True,
    )
    arguments["memory_enabled"] = True
    calls = []
    agent = HypothesisAgent(
        comparison_workers=1,
        model_provider=_catalog_provider(calls),
    )
    retrieval = arguments.pop("initial_retrieval_phase")
    first = agent.run_hypothesis_operation(
        task_root=str(tmp_path),
        operation_instance_id="hypothesis-capability-drift-operation",
        initial_retrieval_phase=retrieval,
        expected_task_id="task-capability-drift",
        expected_batch_id="batch-1",
        **arguments,
    )
    package = copy.deepcopy(arguments["regeneration_package"])
    base_input = copy.deepcopy(package)
    base_input.pop("source_record_ids", None)
    base_input["stage_tool_projection"] = copy.deepcopy(
        arguments["stage_tool_projection"]
    )
    drifted_projection = copy.deepcopy(
        arguments["chemical_capability_projection"]
    )
    drifted_projection[0]["description"] = "漂移后的原子能力描述"
    base_input["chemical_capability_projection"] = drifted_projection
    provider_call_count = len(calls)

    with pytest.raises(ModelOperationError, match="base_input 与 receipt 不一致"):
        agent.replay_hypothesis_operation_from_finalized_manifest(
            task_root=str(tmp_path),
            operation_instance_id="hypothesis-capability-drift-operation",
            base_input=base_input,
            scientific_question=arguments["scientific_question"],
            initial_query_angles=arguments["initial_query_angles"],
            initial_retrieval_phase=first["retrieval_result"],
            chemistry_context=arguments["chemistry_context"],
            num_queries=arguments["num_queries"],
            num_hypotheses_per_query=arguments["num_hypotheses_per_query"],
            top_n=arguments["top_n"],
            hypothesis_ids_by_query=arguments["hypothesis_ids_by_query"],
            merge_hypothesis_ids=arguments["merge_hypothesis_ids"],
            regeneration_package=package,
            memory_enabled=True,
            expected_task_id="task-capability-drift",
            expected_batch_id="batch-1",
        )

    assert len(calls) == provider_call_count


def test_hypothesis_agent_regeneration_adapter_rebuilds_request_from_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hypothesis adapter 只核对 request 摘要，不要求 receipt 保存完整请求正文。"""

    bridge, store, _generator, _writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
    )
    snapshot = store.load_investigation_snapshot("task-1")
    assert snapshot is not None
    request = bridge._build_generation_request_from_state(
        snapshot.state,
        task_id="task-1",
        next_batch_id="batch-2",
        next_batch_index=2,
        operation_key="regenerate:task-1:batch-2",
        operation_instance_id="hypothesis-operation-2",
        generation_config=bridge._test_generation_config,
    )
    started = bridge._create_started_receipt(
        snapshot,
        request=request,
        identities=_identities(),
        source_record_ids=(),
        causal_source_ref=None,
    )
    assert started is not None
    agent = HypothesisAgent(comparison_workers=1)
    replay_calls = []

    def replay(**kwargs):
        replay_calls.append(kwargs)
        return {"replayed": True}

    monkeypatch.setattr(agent, "replay_hypothesis_operation_from_finalized_manifest", replay)
    adapter = HypothesisAgentRegenerationGenerator(agent)

    result = adapter.recover_hypothesis_regeneration(request, started.receipt)

    assert result == {"replayed": True}
    assert len(replay_calls) == 1
    assert replay_calls[0]["base_input"]["chemical_capability_projection"] == list(
        request.generation_config.chemical_capability_projection
    )


def test_hypothesis_agent_regeneration_adapter_rejects_capability_digest_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """started receipt 冻结能力事实后，变更投影不得进入 finalized replay。"""

    bridge, store, _generator, _writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
    )
    snapshot = store.load_investigation_snapshot("task-1")
    assert snapshot is not None
    request = bridge._build_generation_request_from_state(
        snapshot.state,
        task_id="task-1",
        next_batch_id="batch-2",
        next_batch_index=2,
        operation_key="regenerate:task-1:batch-2",
        operation_instance_id="hypothesis-operation-2",
        generation_config=bridge._test_generation_config,
    )
    started = bridge._create_started_receipt(
        snapshot,
        request=request,
        identities=_identities(),
        source_record_ids=(),
        causal_source_ref=None,
    )
    assert started is not None
    projection = copy.deepcopy(
        list(request.generation_config.chemical_capability_projection)
    )
    projection[0]["description"] = "漂移后的原子能力描述"
    drifted_config = replace(
        request.generation_config,
        chemical_capability_projection=tuple(projection),
    )
    drifted_request = replace(request, generation_config=drifted_config)
    agent = HypothesisAgent(comparison_workers=1)
    replay_calls = []

    def replay(**kwargs):
        replay_calls.append(kwargs)
        return {"replayed": True}

    monkeypatch.setattr(
        agent,
        "replay_hypothesis_operation_from_finalized_manifest",
        replay,
    )
    adapter = HypothesisAgentRegenerationGenerator(agent)

    with pytest.raises(ValueError, match="model request 摘要与当前请求不一致"):
        adapter.recover_hypothesis_regeneration(
            drifted_request,
            started.receipt,
        )

    assert replay_calls == []


def test_hypothesis_agent_regeneration_adapter_forwards_generation_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hypothesis adapter 只负责把当前 request 转发给既有 operation。"""

    bridge, store, _generator, _writer = _bridge(
        tmp_path,
        mark_all_refuted(make_initial_state(), "initial"),
    )
    snapshot = store.load_investigation_snapshot("task-1")
    assert snapshot is not None
    request = bridge._build_generation_request_from_state(
        snapshot.state,
        task_id="task-1",
        next_batch_id="batch-2",
        next_batch_index=2,
        operation_key="regenerate:task-1:batch-2",
        operation_instance_id="hypothesis-operation-2",
        generation_config=bridge._test_generation_config,
    )
    agent = HypothesisAgent(comparison_workers=1)
    operation_calls = []

    def generate(**kwargs):
        operation_calls.append(kwargs)
        return {"generated": True}

    monkeypatch.setattr(agent, "run_hypothesis_operation", generate)
    adapter = HypothesisAgentRegenerationGenerator(agent)

    result = adapter.generate_hypothesis_regeneration(request)

    assert result == {"generated": True}
    assert len(operation_calls) == 1
    assert operation_calls[0]["chemical_capability_projection"] == list(
        request.generation_config.chemical_capability_projection
    )
