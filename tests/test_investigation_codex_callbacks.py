"""Investigation Codex 回调组合器的确定性合同测试。"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, replace
from typing import Any, Mapping

import pytest

from chemistry_multiagent.investigation.contracts import (
    InvestigationResponse,
    PassReviewFinding,
    TaskReviewResponse,
)
from chemistry_multiagent.investigation.codex_runtime import InvestigationTurnResult
from chemistry_multiagent.investigation.async_jobs import (
    InvestigationJobIdentity,
    InvestigationJobState,
)
from chemistry_multiagent.investigation.domain import (
    InvestigationResult,
    InvestigationSession,
    InvestigationState,
)
from chemistry_multiagent.investigation.prompts import (
    build_investigation_job_continuation_request,
)


COMBINATION_IDS = tuple(f"combination-{index}" for index in range(1, 6))
_MULTILINE_SCIENTIFIC_QUESTION = (
    "Assess the stereochemical outcome for SMILES "
    "C[C@H](O)/C=C\\C1=CC=CC=C1 at 30\N{DEGREE SIGN}C.\t"
    "Compare the computed result with the specified conditions.\n\n"
    "Experimentally reported enantiomeric excess: 88%."
)


def _source_artifact_ref(combination_id: str, version: str) -> str:
    """返回真实 MiniChem-style 原始 artifact ref。"""

    return "art_" + hashlib.sha256(
        f"source-{combination_id}-{version}".encode()
    ).hexdigest()[:32]


def _reader_artifact_id(source_ref: str) -> str:
    """返回与 MiniChem source ref 分离的 Evidence reader ID。"""

    return "art_" + hashlib.sha256(f"reader-{source_ref}".encode()).hexdigest()[:32]


def _response(combination_id: str, version: str) -> InvestigationResponse:
    """构造最小但完整的调查响应。"""

    return InvestigationResponse(
        self_judgment="supported",
        narrative=f"{combination_id} {version} 的真实计算回顾。",
        job_refs=(f"job-{combination_id}-{version}",),
        operation_refs=(f"operation-{combination_id}-{version}",),
        artifact_refs=(_source_artifact_ref(combination_id, version),),
        observations=(f"observation-{combination_id}-{version}",),
        limitations=(),
    )


def _state(*, batch_id: str = "batch-1", phase: str = "created") -> InvestigationState:
    """构造五个组合的当前批次状态。"""

    sessions = tuple(
        InvestigationSession(
            task_id="task-1",
            batch_id=batch_id,
            combination_id=combination_id,
            codex_thread_id=(
                None if phase == "created" else f"thread-{combination_id}"
            ),
            workspace_ref=f"workspace-{combination_id}",
            scope_id=f"scope-{combination_id}",
            result_version=(None if phase == "created" else "v1"),
            phase=phase,
        )
        for combination_id in COMBINATION_IDS
    )
    results: tuple[InvestigationResult, ...] = ()
    if phase != "created":
        results = tuple(
            InvestigationResult(
                task_id="task-1",
                batch_id=batch_id,
                combination_id=combination_id,
                result_version="v1",
                codex_thread_id=f"thread-{combination_id}",
                scope_id=f"scope-{combination_id}",
                turn_id=f"turn-{combination_id}-v1",
                self_judgment="supported",
                narrative=f"{combination_id} 的结果。",
                job_refs=(f"job-{combination_id}-v1",),
                operation_refs=(f"operation-{combination_id}-v1",),
                artifact_refs=(_source_artifact_ref(combination_id, "v1"),),
                observations=(f"observation-{combination_id}-v1",),
                limitations=(),
            )
            for combination_id in COMBINATION_IDS
        )
    return InvestigationState(sessions=sessions, results=results, task_id="task-1", batch_id=batch_id)


@dataclass
class _FakeRuntime:
    """记录请求和 scope 的假 runtime。"""

    name: str
    calls: list[dict[str, Any]]
    review_calls: list[dict[str, Any]]

    def run_investigation_turn(self, request: Mapping[str, Any], *, scope_id: str) -> InvestigationTurnResult:
        """返回按请求身份生成的调查响应。"""

        self.calls.append({"request": request, "scope_id": scope_id})
        payload = json.loads(request["messages"][1]["content"])
        metadata = request["metadata"]
        execution_input_id = metadata["hypothesis_execution_input_record_id"]
        if metadata["request_kind"] == "initial_investigation":
            version = "v1"
        else:
            version = f"v{int(payload['result_version'][1:]) + 1}"
        return InvestigationTurnResult(
            result=_response(execution_input_id, version),
            thread_id=f"thread-{execution_input_id}",
            turn_id=f"turn-{execution_input_id}-{version}",
        )

    def run_task_review_turn(self, request: Mapping[str, Any], *, scope_id: str) -> InvestigationTurnResult:
        """返回固定五项 pass Review。"""

        self.review_calls.append({"request": request, "scope_id": scope_id})
        return InvestigationTurnResult(
            result=TaskReviewResponse(
                findings=tuple(
                    PassReviewFinding(
                        hypothesis_execution_input_record_id=combination_id,
                        validity="pass",
                        hypothesis_judgment="supported",
                    )
                    for combination_id in COMBINATION_IDS
                ),
                can_answer_original_question=True,
                final_answer="五项结果支持对原科学问题的回答。",
            ),
            thread_id="review-thread-1",
            turn_id=f"review-turn-{len(self.review_calls)}",
        )


@dataclass
class _WakeRuntime:
    """只记录 wake envelope，验证恢复不会重建新的请求。"""

    calls: list[dict[str, Any]]

    def run_investigation_turn(
        self,
        request: Mapping[str, Any],
        *,
        scope_id: str,
        active_job: InvestigationJobState | Mapping[str, Any] | None = None,
    ) -> InvestigationTurnResult:
        """接收完整 request，不解析或改写其模型可见内容。"""

        self.calls.append(
            {"request": request, "scope_id": scope_id, "active_job": active_job}
        )
        return InvestigationTurnResult(
            result=None,
            thread_id=request["thread_id"],
            turn_id="wake-turn-1",
        )


def _snapshot_builder(
    state: InvestigationState,
    results: tuple[InvestigationResult, ...],
    vector: tuple[tuple[str, str], ...],
) -> Mapping[str, Any]:
    """返回严格公开的 evidence manifest。"""

    return {
        "snapshot_id": f"snapshot-{state.batch_id}-{len(state.validations)}",
        "task_id": state.task_id,
        "batch_id": state.batch_id,
        "result_pairs": [
            {
                "combination_id": combination_id,
                "result_version": version,
                "artifacts": [
                    {
                        "artifact_id": _reader_artifact_id(result.artifact_refs[0]),
                        "source_artifact_ref": result.artifact_refs[0],
                        "size": 1,
                        "sha256": hashlib.sha256(
                            f"reader-{combination_id}".encode()
                        ).hexdigest(),
                        "media_type": "application/json",
                    }
                ],
            }
            for (combination_id, version), result in zip(vector, results)
        ],
    }


def _config(
    *,
    runtimes: Mapping[tuple[str, str, str], _FakeRuntime] | None = None,
    scientific_question: str = "判断科学问题。",
):
    """构造 callback 配置；导入放在 helper 内以保持测试边界清晰。"""

    from chemistry_multiagent.investigation.codex_callbacks import (
        InvestigationCodexCallbackConfig,
        InvestigationCombinationInput,
        InvestigationSessionIdentity,
    )

    inputs = {
        InvestigationSessionIdentity("task-1", "batch-1", combination_id): InvestigationCombinationInput(
            hypothesis={"strategy": combination_id},
            explicit_conditions={"temperature": 300},
            frozen_context={"fact": combination_id},
            working_memory={"memory": combination_id},
            calculation_unit={"reactants": [combination_id]},
            available_tools=("read_artifact",),
            tool_descriptors=({"name": "read_artifact", "kind": "readonly"},),
        )
        for combination_id in COMBINATION_IDS
    }
    runtime_values = runtimes or {
        identity: _FakeRuntime(identity.combination_id, [], [])
        for identity in inputs
    }
    return InvestigationCodexCallbackConfig(
        task_id="task-1",
        scientific_question=scientific_question,
        inputs_by_session=inputs,
        runtimes_by_session=runtime_values,
        review_runtime=_FakeRuntime("review", [], []),
        review_scope_id="review-scope",
        evidence_snapshot_builder=_snapshot_builder,
        evidence_reader_tool={"name": "read_snapshot_artifact", "kind": "readonly_evidence_reader"},
        result_verifier=lambda _session, result: result,
        investigation_model="investigation-model",
        review_model="review-model",
    )


def test_callback_config_accepts_multiline_scientific_question() -> None:
    """Codex callback 配置允许公开科学正文中的换行、制表符和 SMILES。"""

    config = _config(scientific_question=_MULTILINE_SCIENTIFIC_QUESTION)

    assert config.scientific_question == _MULTILINE_SCIENTIFIC_QUESTION


def test_callbacks_build_frozen_per_session_requests_and_verify_initial_results() -> None:
    """五个 identity 选到各自输入/runtime，首次结果固定为 v1 并经过 verifier。"""

    from chemistry_multiagent.investigation.codex_callbacks import InvestigationCodexCallbacks

    config = _config()
    verified: list[str] = []
    object.__setattr__(
        config,
        "result_verifier",
        lambda session, result: (verified.append(session.combination_id) or result),
    )
    callbacks = InvestigationCodexCallbacks(config)
    state = _state()
    results = [callbacks.run_initial_investigation(session) for session in state.sessions]

    assert [result.result_version for result in results] == ["v1"] * 5
    assert verified == list(COMBINATION_IDS)
    for session, result in zip(state.sessions, results):
        runtime = config.runtimes_by_session[
            (session.task_id, session.batch_id, session.combination_id)
        ]
        assert runtime.calls[0]["scope_id"] == session.scope_id
        payload = json.loads(runtime.calls[0]["request"]["messages"][1]["content"])
        assert payload["hypothesis"] == {"strategy": session.combination_id}
        assert payload["explicit_conditions"] == {"temperature": 300}
        assert payload["frozen_context"] == {"fact": session.combination_id}
        assert "working_memory" not in payload
        assert payload["calculation_unit"] == {"reactants": [session.combination_id]}
        assert runtime.calls[0]["request"]["model"] == "investigation-model"

    with pytest.raises(TypeError):
        config.inputs_by_session[
            ("task-1", "batch-1", "combination-1")
        ] = config.inputs_by_session[("task-1", "batch-1", "combination-1")]


def test_initial_investigation_does_not_expose_cross_batch_working_memory() -> None:
    """首次调查不能把跨批次科学经验投影到模型 user JSON。"""

    from chemistry_multiagent.investigation.codex_callbacks import InvestigationCodexCallbacks

    config = _config()
    original = config.inputs_by_session[("task-1", "batch-1", "combination-1")]
    replacement = replace(
        original,
        working_memory={
            "closed_batches": [
                {"scientific_experience": "CROSS_BATCH_EXPERIENCE_MUST_STAY_PRIVATE"}
            ]
        },
    )
    inputs = dict(config.inputs_by_session)
    inputs[("task-1", "batch-1", "combination-1")] = replacement
    config = replace(config, inputs_by_session=inputs)

    callbacks = InvestigationCodexCallbacks(config)
    callbacks.run_initial_investigation(_state().sessions[0])

    request = config.runtimes_by_session[("task-1", "batch-1", "combination-1")].calls[0][
        "request"
    ]
    payload_text = request["messages"][1]["content"]
    payload = json.loads(payload_text)
    assert "working_memory" not in payload
    assert "CROSS_BATCH_EXPERIENCE_MUST_STAY_PRIVATE" not in payload_text


def test_callbacks_repair_reuses_thread_and_deterministic_version() -> None:
    """返修请求只携带当前版本/原建议，并按 repair_count 产生新版本。"""

    from chemistry_multiagent.investigation.codex_callbacks import InvestigationCodexCallbacks

    config = _config()
    callbacks = InvestigationCodexCallbacks(config)
    session = _state(phase="repair_pending").sessions[0].model_copy(
        update={"repair_count": 1, "result_version": "v2"}
    )
    result = callbacks.run_deepseek_repair(session, "原样保留的返修建议")
    runtime = config.runtimes_by_session[("task-1", "batch-1", "combination-1")]
    payload = json.loads(runtime.calls[0]["request"]["messages"][1]["content"])

    assert result.result_version == "v3"
    assert result.codex_thread_id == session.codex_thread_id
    assert runtime.calls[0]["scope_id"] == session.scope_id
    assert runtime.calls[0]["request"]["thread_id"] == session.codex_thread_id
    assert payload["result_version"] == "v2"
    assert payload["repair_advice"] == "原样保留的返修建议"


def test_callbacks_repair_recovery_rebuilds_the_original_repair_request() -> None:
    """repair_pending recovery 必须把当前建议重建为 deepseek_repair request。"""

    from chemistry_multiagent.investigation.codex_callbacks import InvestigationCodexCallbacks

    config = _config()
    runtime = config.runtimes_by_session[("task-1", "batch-1", "combination-1")]
    recovery_calls: list[dict[str, Any]] = []
    runtime.has_recoverable_interrupted_investigation_turn = (  # type: ignore[attr-defined]
        lambda request, *, scope_id, active_job: (
            recovery_calls.append(
                {
                    "request": request,
                    "scope_id": scope_id,
                    "active_job": active_job,
                }
            )
            or True
        )
    )
    runtime.recover_interrupted_investigation_turn = (  # type: ignore[attr-defined]
        lambda request, *, scope_id, active_job: recovery_calls.append(
            {
                "request": request,
                "scope_id": scope_id,
                "active_job": active_job,
                "recovered": True,
            }
        )
    )
    callbacks = InvestigationCodexCallbacks(config)
    session = _state(phase="repair_pending").sessions[0]
    waiting = InvestigationJobState(
        task_id=session.task_id,
        batch_id=session.batch_id,
        combination_id=session.combination_id,
        codex_thread_id=session.codex_thread_id,
        turn_id="repair-turn-1",
        status="waiting_investigation_job",
        identity=InvestigationJobIdentity(
            job_id="job-repair-1",
            operation_id="operation-repair-1",
            submission_key="submission-repair-1",
            action_id="action-repair-1",
            backend_id="backend-repair-1",
        ),
        request_digest="a" * 64,
        scope_id=session.scope_id,
    )

    assert callbacks.has_recoverable_interrupted_investigation_turn(
        session,
        waiting,
        repair_advice="原样保留的返修建议",
    ) is True
    callbacks.recover_interrupted_investigation_turn(
        session,
        waiting,
        repair_advice="原样保留的返修建议",
    )

    assert len(recovery_calls) == 2
    for call in recovery_calls:
        request = call["request"]
        assert request["metadata"]["request_kind"] == "deepseek_repair"
        assert request["thread_id"] == session.codex_thread_id
        payload = json.loads(request["messages"][1]["content"])
        assert payload["result_version"] == "v1"
        assert payload["repair_advice"] == "原样保留的返修建议"
        assert call["scope_id"] == session.scope_id
        assert call["active_job"] is waiting


def test_callbacks_repair_continuation_recovery_reuses_continuation_request_builder() -> None:
    """后续 repair attempt 的 recovery request 必须与正常 continuation 完全一致。"""

    from chemistry_multiagent.investigation.codex_callbacks import InvestigationCodexCallbacks

    config = _config()
    callbacks = InvestigationCodexCallbacks(config)
    session = _state(phase="repair_pending").sessions[0]
    waiting = InvestigationJobState(
        task_id=session.task_id,
        batch_id=session.batch_id,
        combination_id=session.combination_id,
        codex_thread_id=session.codex_thread_id,
        turn_id="repair-continuation-turn",
        status="waiting_investigation_job",
        identity=InvestigationJobIdentity(
            job_id="job-repair-continuation",
            operation_id="operation-repair-continuation",
            submission_key="submission-repair-continuation",
            action_id="search_transition_state_path",
            backend_id="transition_state_path_search",
        ),
        request_digest="a" * 64,
        scope_id=session.scope_id,
    )
    runtime = config.runtimes_by_session[("task-1", "batch-1", "combination-1")]
    recovery_calls: list[dict[str, Any]] = []
    runtime.has_recoverable_interrupted_investigation_turn = (  # type: ignore[attr-defined]
        lambda request, *, scope_id, active_job: (
            recovery_calls.append(
                {
                    "request": request,
                    "scope_id": scope_id,
                    "active_job": active_job,
                }
            )
            or True
        )
    )
    runtime.recover_interrupted_investigation_turn = (  # type: ignore[attr-defined]
        lambda request, *, scope_id, active_job: recovery_calls.append(
            {
                "request": request,
                "scope_id": scope_id,
                "active_job": active_job,
                "recovered": True,
            }
        )
    )
    inputs = config.inputs_by_session[("task-1", "batch-1", "combination-1")]
    expected = build_investigation_job_continuation_request(
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
        scientific_question=config.scientific_question,
        hypothesis=dict(inputs.hypothesis),
        explicit_conditions=dict(inputs.explicit_conditions),
        frozen_context=dict(inputs.frozen_context),
        calculation_unit=dict(inputs.calculation_unit),
        available_tools=(
            *inputs.available_tools,
            waiting.submit_tool,
            waiting.status_tool,
            waiting.collect_tool,
            "list_execution_jobs",
        ),
        tool_descriptors=tuple(dict(descriptor) for descriptor in inputs.tool_descriptors),
        model=config.investigation_model,
    )

    assert callbacks.has_recoverable_interrupted_investigation_turn(
        session,
        waiting,
        request_kind="investigation_job_continuation",
    ) is True
    callbacks.recover_interrupted_investigation_turn(
        session,
        waiting,
        request_kind="investigation_job_continuation",
    )

    assert len(recovery_calls) == 2
    assert recovery_calls[0]["request"] == expected
    assert recovery_calls[1]["request"] == expected
    assert expected["metadata"]["request_kind"] == "investigation_job_continuation"
    payload = json.loads(expected["messages"][-1]["content"])
    assert "repair_advice" not in payload
    assert recovery_calls[0]["active_job"] is waiting


def test_callbacks_replay_persisted_wake_envelope_without_rebuilding_prompt() -> None:
    """恢复 wake 直接复用 sidecar envelope，Prompt 版本变化不会产生新请求。"""

    from chemistry_multiagent.investigation.codex_callbacks import InvestigationCodexCallbacks
    from chemistry_multiagent.investigation.prompts import build_investigation_job_wake_request

    base_config = _config()
    session = _state(phase="repair_pending").sessions[0]
    identity = InvestigationJobIdentity(
        job_id="job_0123456789abcdef0123456789abcdef",
        operation_id="operation-wake-1",
        submission_key="submission-wake-1",
        action_id="search_transition_state_path",
        backend_id="transition_state_path_search",
    )
    waiting = InvestigationJobState(
        task_id=session.task_id,
        batch_id=session.batch_id,
        combination_id=session.combination_id,
        codex_thread_id=session.codex_thread_id,
        turn_id="turn-submit-1",
        status="waiting_investigation_job",
        identity=identity,
        request_digest="a" * 64,
        scope_id=session.scope_id,
    )
    exact_request = build_investigation_job_wake_request(
        session.task_id,
        session.batch_id,
        session.combination_id,
        session.codex_thread_id,
        job_id=identity.job_id,
        operation_id=identity.operation_id,
        submission_key=identity.submission_key,
        action_id=identity.action_id,
        backend_id=identity.backend_id,
        terminal_status="succeeded",
        available_tools=(
            "read_artifact",
            "submit_transition_state_path_job",
            "get_transition_state_path_job",
            "collect_transition_state_path_job",
        ),
        scientific_question=base_config.scientific_question,
        hypothesis={"strategy": "combination-1"},
        explicit_conditions={"temperature": 300},
        frozen_context={"fact": "combination-1"},
        calculation_unit={"reactants": ["combination-1"]},
        model="investigation-model",
    )
    wake_runtime = _WakeRuntime([])
    runtimes = dict(base_config.runtimes_by_session)
    runtimes[(session.task_id, session.batch_id, session.combination_id)] = wake_runtime
    callbacks = InvestigationCodexCallbacks(
        replace(base_config, runtimes_by_session=runtimes)
    )

    callbacks.run_investigation_job_wake(
        session,
        waiting,
        "succeeded",
        wake_request=exact_request,
    )

    assert wake_runtime.calls[0]["request"] == exact_request
    assert wake_runtime.calls[0]["request"] is not exact_request
    assert wake_runtime.calls[0]["active_job"] is waiting


@pytest.mark.parametrize(
    "identity_field",
    [
        "job_family",
        "action_id",
        "backend_id",
        "job_type",
        "runtime",
        "software_id",
        "operation_fingerprint",
    ],
)
def test_callbacks_reject_persisted_action_wake_identity_drift(
    identity_field: str,
) -> None:
    """持久 wake 不能在保留三项作业 ID 时替换 action_execution 身份。"""

    from chemistry_multiagent.investigation.codex_callbacks import InvestigationCodexCallbacks
    from chemistry_multiagent.investigation.prompts import build_investigation_job_wake_request

    base_config = _config()
    session = _state(phase="repair_pending").sessions[0]
    identity = InvestigationJobIdentity(
        job_id="job_action_persisted_wake",
        operation_id="operation-action-persisted-wake",
        submission_key="submission-action-persisted-wake",
        action_id="scan_reaction_coordinates",
        backend_id="gaussian",
        job_family="action_execution",
        job_type="action_execution",
        runtime="gaussian",
        software_id="gaussian",
        operation_fingerprint="a" * 64,
    )
    waiting = InvestigationJobState(
        task_id=session.task_id,
        batch_id=session.batch_id,
        combination_id=session.combination_id,
        codex_thread_id=session.codex_thread_id,
        turn_id="turn-action-persisted-wake",
        status="waiting_investigation_job",
        identity=identity,
        request_digest="b" * 64,
        scope_id=session.scope_id,
        job_family="action_execution",
        submit_tool="submit_action_job",
        status_tool="get_execution_job",
        collect_tool="collect_execution_job",
    )
    request = build_investigation_job_wake_request(
        session.task_id,
        session.batch_id,
        session.combination_id,
        session.codex_thread_id,
        job_id=identity.job_id,
        operation_id=identity.operation_id,
        submission_key=identity.submission_key,
        action_id=identity.action_id,
        backend_id=identity.backend_id,
        job_family=identity.job_family,
        job_type=identity.job_type,
        runtime=identity.runtime,
        software_id=identity.software_id,
        operation_fingerprint=identity.operation_fingerprint,
        scope_id=waiting.scope_id,
        terminal_status="succeeded",
        available_tools=(
            waiting.status_tool,
            waiting.collect_tool,
        ),
        model="investigation-model",
    )
    request["metadata"][identity_field] = (
        "native_execution"
        if identity_field == "job_family"
        else f"different-{identity_field}"
    )

    with pytest.raises(ValueError, match="作业身份|与等待状态不一致"):
        InvestigationCodexCallbacks._validate_persisted_wake_request(
            request,
            session=session,
            waiting=waiting,
            terminal_status="succeeded",
        )


def test_callbacks_upgrade_persisted_minimal_wake_without_combination_drift() -> None:
    """旧的最小 wake envelope 升级为完整上下文且不漂移 combination identity。"""

    from chemistry_multiagent.investigation.codex_callbacks import InvestigationCodexCallbacks
    from chemistry_multiagent.investigation.prompts import build_investigation_job_wake_request

    base_config = _config()
    session = _state(phase="repair_pending").sessions[0]
    identity = InvestigationJobIdentity(
        job_id="job_0123456789abcdef0123456789abcdef",
        operation_id="operation-wake-legacy-1",
        submission_key="submission-wake-legacy-1",
        action_id="search_transition_state_path",
        backend_id="transition_state_path_search",
    )
    waiting = InvestigationJobState(
        task_id=session.task_id,
        batch_id=session.batch_id,
        combination_id=session.combination_id,
        codex_thread_id=session.codex_thread_id,
        turn_id="turn-submit-legacy-1",
        status="waiting_investigation_job",
        identity=identity,
        request_digest="b" * 64,
        scope_id=session.scope_id,
    )
    minimal_request = build_investigation_job_wake_request(
        session.task_id,
        session.batch_id,
        session.combination_id,
        session.codex_thread_id,
        job_id=identity.job_id,
        operation_id=identity.operation_id,
        submission_key=identity.submission_key,
        action_id=identity.action_id,
        backend_id=identity.backend_id,
        terminal_status="succeeded",
        available_tools=(
            waiting.status_tool,
            waiting.collect_tool,
        ),
        model="investigation-model",
    )
    wake_runtime = _WakeRuntime([])
    runtimes = dict(base_config.runtimes_by_session)
    runtimes[(session.task_id, session.batch_id, session.combination_id)] = wake_runtime
    callbacks = InvestigationCodexCallbacks(
        replace(base_config, runtimes_by_session=runtimes)
    )

    callbacks.run_investigation_job_wake(
        session,
        waiting,
        "succeeded",
        wake_request=minimal_request,
    )

    upgraded = wake_runtime.calls[0]["request"]
    assert wake_runtime.calls[0]["active_job"] is waiting
    payload = json.loads(upgraded["messages"][-1]["content"])
    assert (
        upgraded["metadata"]["hypothesis_execution_input_record_id"]
        == session.combination_id
    )
    assert payload["scientific_question"] == base_config.scientific_question
    assert payload["hypothesis"] == {"strategy": "combination-1"}
    assert payload["explicit_conditions"] == {"temperature": 300}
    assert payload["frozen_context"] == {"fact": "combination-1"}
    assert payload["calculation_unit"] == {"reactants": ["combination-1"]}


def test_review_uses_one_task_runtime_and_continuous_thread_without_minichem() -> None:
    """Review 使用固定 task scope/runtime，完整读取五项快照并跨批次带 thread。"""

    from chemistry_multiagent.investigation.codex_callbacks import InvestigationCodexCallbacks

    config = _config()
    callbacks = InvestigationCodexCallbacks(config)
    review_runtime = config.review_runtime
    first_state = _state(phase="ready_for_batch_validation")
    first_vector = tuple((item.combination_id, item.result_version) for item in first_state.sessions)
    first = callbacks.run_task_batch_review(first_state, first_state.results, first_vector)
    second_state = first_state.model_copy(update={"validations": (first,)})
    # 恢复测试使用相同 callback 只验证已有批次验证记录带来的 task 级 thread 连续性。
    second_vector = tuple((item.combination_id, item.result_version) for item in second_state.sessions)
    second = callbacks.run_task_batch_review(second_state, second_state.results, second_vector)

    assert first.review_thread_id == second.review_thread_id == "review-thread-1"
    assert len(review_runtime.review_calls) == 2
    assert [call["scope_id"] for call in review_runtime.review_calls] == ["review-scope"] * 2
    assert review_runtime.review_calls[1]["request"]["thread_id"] == "review-thread-1"
    for call in review_runtime.review_calls:
        payload = json.loads(call["request"]["messages"][1]["content"])
        assert len(payload["current_results"]) == 5
        assert "tools" in call["request"]
        assert "minichem" not in json.dumps(call["request"]["tools"], ensure_ascii=False).lower()


def test_callbacks_fail_closed_on_identity_vector_snapshot_and_verifier_errors() -> None:
    """session/vector/snapshot 漂移和 verifier 异常不得被吞掉。"""

    from chemistry_multiagent.investigation.codex_callbacks import InvestigationCodexCallbacks

    config = _config()
    callbacks = InvestigationCodexCallbacks(config)
    with pytest.raises(ValueError, match="identity"):
        callbacks.run_initial_investigation(
            _state().sessions[0].model_copy(update={"task_id": "other-task"})
        )

    state = _state(phase="ready_for_batch_validation")
    vector = tuple((item.combination_id, item.result_version) for item in state.sessions)
    with pytest.raises(ValueError, match="vector"):
        callbacks.run_task_batch_review(
            state,
            state.results,
            tuple((combination_id, "v999") for combination_id, _ in vector),
        )

    def bad_snapshot(*_args: Any) -> Mapping[str, Any]:
        """返回不属于任务的快照。"""

        return {
            "snapshot_id": "snapshot-bad",
            "task_id": "other-task",
            "batch_id": "batch-1",
            "result_pairs": [],
        }

    from chemistry_multiagent.investigation.codex_callbacks import InvestigationCodexCallbackConfig

    bad_config = InvestigationCodexCallbackConfig(
        task_id=config.task_id,
        scientific_question=config.scientific_question,
        inputs_by_session=config.inputs_by_session,
        runtimes_by_session=config.runtimes_by_session,
        review_runtime=config.review_runtime,
        review_scope_id=config.review_scope_id,
        evidence_snapshot_builder=bad_snapshot,
        evidence_reader_tool=config.evidence_reader_tool,
        result_verifier=config.result_verifier,
        investigation_model=config.investigation_model,
        review_model=config.review_model,
    )
    with pytest.raises(ValueError):
        InvestigationCodexCallbacks(bad_config).run_task_batch_review(state, state.results, vector)

    def rejecting_verifier(_session: InvestigationSession, _result: InvestigationResult) -> None:
        """拒绝未经终态核对的结果。"""

        raise RuntimeError("result is not terminal")

    rejected_config = InvestigationCodexCallbackConfig(
        task_id=config.task_id,
        scientific_question=config.scientific_question,
        inputs_by_session=config.inputs_by_session,
        runtimes_by_session=config.runtimes_by_session,
        review_runtime=config.review_runtime,
        review_scope_id=config.review_scope_id,
        evidence_snapshot_builder=config.evidence_snapshot_builder,
        evidence_reader_tool=config.evidence_reader_tool,
        result_verifier=rejecting_verifier,
        investigation_model=config.investigation_model,
        review_model=config.review_model,
    )
    with pytest.raises(RuntimeError, match="terminal"):
        InvestigationCodexCallbacks(rejected_config).run_initial_investigation(_state().sessions[0])


def test_callbacks_require_verifier_and_reject_shared_runtime_or_missing_mapping() -> None:
    """没有 verifier、重复 runtime 或 identity 映射缺口都在构造/调用时失败。"""

    from chemistry_multiagent.investigation.codex_callbacks import (
        InvestigationCodexCallbackConfig,
        InvestigationCodexCallbacks,
    )

    config = _config()
    with pytest.raises(ValueError, match="verifier"):
        InvestigationCodexCallbackConfig(
            task_id=config.task_id,
            scientific_question=config.scientific_question,
            inputs_by_session=config.inputs_by_session,
            runtimes_by_session=config.runtimes_by_session,
            review_runtime=config.review_runtime,
            review_scope_id=config.review_scope_id,
            evidence_snapshot_builder=config.evidence_snapshot_builder,
            evidence_reader_tool=config.evidence_reader_tool,
            result_verifier=None,
            investigation_model=config.investigation_model,
            review_model=config.review_model,
        )

    shared = _FakeRuntime("shared", [], [])
    shared_runtimes = {key: shared for key in config.inputs_by_session}
    with pytest.raises(ValueError, match="runtime"):
        InvestigationCodexCallbackConfig(
            task_id=config.task_id,
            scientific_question=config.scientific_question,
            inputs_by_session=config.inputs_by_session,
            runtimes_by_session=shared_runtimes,
            review_runtime=config.review_runtime,
            review_scope_id=config.review_scope_id,
            evidence_snapshot_builder=config.evidence_snapshot_builder,
            evidence_reader_tool=config.evidence_reader_tool,
            result_verifier=config.result_verifier,
            investigation_model=config.investigation_model,
            review_model=config.review_model,
        )

    missing = dict(config.runtimes_by_session)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="mapping"):
        InvestigationCodexCallbackConfig(
            task_id=config.task_id,
            scientific_question=config.scientific_question,
            inputs_by_session=config.inputs_by_session,
            runtimes_by_session=missing,
            review_runtime=config.review_runtime,
            review_scope_id=config.review_scope_id,
            evidence_snapshot_builder=config.evidence_snapshot_builder,
            evidence_reader_tool=config.evidence_reader_tool,
            result_verifier=config.result_verifier,
            investigation_model=config.investigation_model,
            review_model=config.review_model,
        )



def test_callbacks_repair_rejects_invalid_phase_thread_version_and_budget() -> None:
    """返修入口拒绝错误阶段、缺失 thread、非法版本和超过三次返修。"""

    from chemistry_multiagent.investigation.codex_callbacks import InvestigationCodexCallbacks

    callbacks = InvestigationCodexCallbacks(_config())
    repair_session = _state(phase="repair_pending").sessions[0]
    invalid_cases = (
        (_state().sessions[0], "repair_pending"),
        (repair_session.model_copy(update={"codex_thread_id": None}), "必须绑定"),
        (repair_session.model_copy(update={"result_version": None}), "必须绑定"),
        (repair_session.model_copy(update={"result_version": "version-2"}), "格式非法"),
        (
            repair_session.model_copy(update={"result_version": "v4", "repair_count": 3}),
            "超过返修上限",
        ),
        (
            repair_session.model_copy(update={"result_version": "v2", "repair_count": 0}),
            "repair_count",
        ),
    )
    for session, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            callbacks.run_deepseek_repair(session, "补足可核对证据")


def test_callbacks_repair_fence_reuses_current_job_and_operation_inventory() -> None:
    """返修 scope fence 只接受当前结果的 job/operation 引用，并保持同一 thread。"""

    from chemistry_multiagent.investigation.codex_callbacks import InvestigationCodexCallbacks

    config = _config()
    fence_calls: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []

    def fence(session: InvestigationSession, jobs, operations) -> Mapping[str, Any]:
        """记录当前 session 的只读副作用清单核对。"""

        fence_calls.append((session.combination_id, tuple(jobs), tuple(operations)))
        return {"ok": True}

    fenced = replace(
        config,
        scope_execution_inventory_fence=fence,
        repair_result_reference_provider=lambda _session: (
            ("job-existing",),
            ("operation-existing",),
        ),
    )
    callbacks = InvestigationCodexCallbacks(fenced)
    callbacks.run_initial_investigation(_state().sessions[0])
    repaired = callbacks.run_deepseek_repair(
        _state(phase="repair_pending").sessions[0].model_copy(
            update={"repair_count": 1, "result_version": "v2"}
        ),
        "补足可核对证据",
    )

    assert repaired.result_version == "v3"
    assert repaired.codex_thread_id == "thread-combination-1"
    assert fence_calls == [
        ("combination-1", (), ()),
        ("combination-1", ("job-existing",), ("operation-existing",)),
    ]

    invalid_references = (
        ("jobs", "ops"),
        (("job-existing",),),
        None,
    )
    for references in invalid_references:
        invalid_config = replace(
            fenced,
            repair_result_reference_provider=lambda _session, references=references: references,
        )
        with pytest.raises(ValueError, match="引用"):
            InvestigationCodexCallbacks(invalid_config).run_deepseek_repair(
                _state(phase="repair_pending").sessions[0].model_copy(
                    update={"repair_count": 1, "result_version": "v2"}
                ),
                "补足可核对证据",
            )


def test_callbacks_config_rejects_missing_or_unsafe_protocol_dependencies() -> None:
    """配置构造器对回调、证据租约、映射和 Review runtime 全部 fail-closed。"""

    config = _config()
    input_values = dict(config.inputs_by_session)
    input_values.pop(next(iter(input_values)))
    invalid_cases = (
        ({"evidence_snapshot_builder": object()}, TypeError, "evidence_snapshot_builder"),
        ({"result_verifier": "not-callable"}, ValueError, "verifier"),
        ({"scope_execution_inventory_fence": object()}, TypeError, "scope_execution_inventory_fence"),
        ({"repair_result_reference_provider": object()}, TypeError, "repair_result_reference_provider"),
        ({"review_thread_id_provider": object()}, TypeError, "review_thread_id_provider"),
        (
            {"scope_execution_inventory_fence": lambda *_args: {}},
            ValueError,
            "repair_result_reference_provider",
        ),
        ({"evidence_lease": object()}, TypeError, "evidence_lease"),
        ({"evidence_reader_tool": []}, TypeError, "JSON object"),
        ({"evidence_reader_tool": {}}, ValueError, "不能为空"),
        ({"review_runtime": object()}, TypeError, "run_task_review_turn"),
        ({"task_id": "other-task"}, ValueError, "task_id"),
        ({"inputs_by_session": input_values}, ValueError, "mapping"),
    )
    for updates, error_type, message in invalid_cases:
        with pytest.raises(error_type, match=message):
            replace(config, **updates)



def test_callbacks_review_result_ordering_rejects_missing_drift_and_extra_facts() -> None:
    """Review 结果排序拒绝缺项、版本/snapshot 事实漂移、scope/thread 漂移和额外组合。"""

    from types import SimpleNamespace

    from chemistry_multiagent.investigation.codex_callbacks import InvestigationCodexCallbacks

    state = _state(phase="ready_for_batch_validation")
    results_by_id = {result.combination_id: result for result in state.results}
    order_results = InvestigationCodexCallbacks._order_review_results

    missing = dict(results_by_id)
    missing.pop("combination-1")
    with pytest.raises(ValueError, match="未完整覆盖"):
        order_results(state, missing)

    version_drift = dict(results_by_id)
    version_drift["combination-1"] = version_drift["combination-1"].model_copy(
        update={"result_version": "v2"}
    )
    with pytest.raises(ValueError, match="result_version 漂移"):
        order_results(state, version_drift)

    saved_fact_drift = dict(results_by_id)
    saved_fact_drift["combination-1"] = saved_fact_drift["combination-1"].model_copy(
        update={"turn_id": "different-turn"}
    )
    with pytest.raises(ValueError, match="已保存"):
        order_results(state, saved_fact_drift)

    scope_drift_state = SimpleNamespace(
        sessions=(state.sessions[0].model_copy(update={"scope_id": "scope-other"}), *state.sessions[1:]),
        results=state.results,
    )
    with pytest.raises(ValueError, match="scope 漂移"):
        order_results(scope_drift_state, results_by_id)

    thread_drift_state = SimpleNamespace(
        sessions=(state.sessions[0].model_copy(update={"codex_thread_id": "thread-other"}), *state.sessions[1:]),
        results=state.results,
    )
    with pytest.raises(ValueError, match="Codex thread 漂移"):
        order_results(thread_drift_state, results_by_id)

    extra = dict(results_by_id)
    extra["combination-extra"] = state.results[0]
    with pytest.raises(ValueError, match="之外"):
        order_results(state, extra)
