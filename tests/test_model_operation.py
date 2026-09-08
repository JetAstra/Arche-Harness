import copy
import hashlib
import json
import threading
from pathlib import Path

import pytest

from chemistry_multiagent.utils import api_retry
from chemistry_multiagent.utils import model_operation


@pytest.fixture(autouse=True)
def _disable_real_retry_sleep(monkeypatch):
    """模型操作失败路径只验证次数，禁止测试真实等待十秒。"""

    monkeypatch.setattr(api_retry.time, "sleep", lambda _delay: None)


_ABSOLUTE_PRIVATE_EVIDENCE = str(
    Path(Path.cwd().anchor) / "tmp" / "private-evidence"
)


def _envelope(content, *, finish_reason="stop", trace_id="trace-1"):
    return {
        "id": trace_id,
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {
                    "content": content,
                    "reasoning_content": "",
                },
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }


def _json_stage(*, domain_validator=None, domain_validator_id=None):
    return model_operation.ModelStage(
        stage="hypothesis",
        stage_kind="model",
        validation_kind="json_schema",
        query_index=1,
        schema={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        },
        domain_validator=domain_validator,
        domain_validator_id=(
            domain_validator_id
            if domain_validator_id is not None
            else ("test.integer.v1" if domain_validator is not None else None)
        ),
    )


def test_domain_validator_requires_a_stable_identity():
    with pytest.raises(
        model_operation.ModelOperationError,
        match="domain_validator_id",
    ):
        model_operation.ModelStage(
            stage="router",
            stage_kind="model",
            validation_kind="json_schema",
            schema={"type": "object"},
            domain_validator=lambda value: value,
        )


def test_fixed_and_dynamic_runners_expose_only_their_workflow_apis(tmp_path):
    """两个公开 runner 必须同级继承，并且不能互相暴露工作流入口。"""

    fixed = model_operation.ModelOperationRunner
    dynamic = model_operation.DynamicModelOperationRunner
    executor = model_operation._ModelStageExecutor

    assert fixed.__bases__ == (executor,)
    assert dynamic.__bases__ == (executor,)
    assert hasattr(fixed, "run_model_operation")
    assert hasattr(fixed, "from_failed_manifest")
    assert hasattr(fixed, "replay_manifest")
    assert not hasattr(fixed, "run")
    assert not hasattr(fixed, "finalize")
    assert not hasattr(fixed, "retry_failed_operation")
    assert not hasattr(fixed, "replay_finalized_operation")
    assert hasattr(dynamic, "run_model_stage")
    assert hasattr(dynamic, "run_reserved_model_stage")
    assert hasattr(dynamic, "run_retrieval_stage")
    assert hasattr(dynamic, "finalize_model_operation")
    assert hasattr(dynamic, "from_failed_manifest")
    assert not hasattr(dynamic, "run")
    assert not hasattr(dynamic, "run_model_operation")
    assert not hasattr(dynamic, "finalize")
    assert not hasattr(dynamic, "retry_failed_operation")
    assert not hasattr(dynamic, "replay_finalized_operation")
    assert not hasattr(dynamic, "replay_manifest")

    runner = dynamic(
        task_root=tmp_path,
        operation_instance_id="dynamic-api-boundary",
        operation="hypothesis_generation",
        base_input={"scientific_question": "验证动态入口边界。"},
        provider=lambda _request: pytest.fail("没有预留 stage 时不应调用模型"),
        message_builder=_builder([]),
        transport_attempts=1,
    )
    with pytest.raises(model_operation.ModelOperationError, match="未完成的预留 stage"):
        runner.finalize_model_operation()
    assert list(runner.operation_dir.glob("manifest_*.json")) == []


def test_runner_persists_the_exact_compiled_request_sent_to_provider(tmp_path):
    observed = []

    def provider(request):
        observed.append(copy.deepcopy(request))
        return _envelope('{"value":7}')

    runner = model_operation.ModelOperationRunner(
        task_root=tmp_path,
        operation_instance_id="compiled-request",
        operation="hypothesis_generation",
        base_input={"scientific_question": "验证编译请求。"},
        stages=[_json_stage()],
        provider=provider,
        message_builder=_builder([]),
        transport_attempts=1,
    )

    result = runner.run_model_operation()

    assert result["status"] == "validated"
    assert len(observed) == 1
    assert observed[0]["catalog_key"] == "hypothesis_generation.hypothesis"
    assert observed[0]["max_tokens"] is None
    assert len(observed[0]["response_schema_hash"]) == 64
    attempt = result["manifest"]["stages"][0]["attempts"][0]
    assert _read_ref(tmp_path, attempt["request_ref"]) == observed[0]


def _builder(call_log):
    def build(
        operation,
        base_input,
        stage=None,
        protocol_error=None,
        memory_enabled=None,
        retry_failed_operation=False,
        expert_fallback=False,
    ):
        call_log.append(
            {
                "operation": operation,
                "base_input": base_input,
                "stage": stage,
                "protocol_error": protocol_error,
                "retry_failed_operation": retry_failed_operation,
            }
        )
        return [
            {"role": "system", "content": "固定协议"},
            {"role": "user", "content": json.dumps(base_input, sort_keys=True)},
        ]

    return build


def _runner(
    tmp_path,
    responses,
    stages,
    *,
    operation_instance_id="operation-1",
    transport_attempts=3,
    call_log=None,
):
    provider_calls = []
    response_queue = list(responses)

    def provider(messages):
        provider_calls.append(messages)
        response = response_queue.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    calls = call_log if call_log is not None else []
    runner = model_operation.ModelOperationRunner(
        task_root=tmp_path,
        operation_instance_id=operation_instance_id,
        operation="hypothesis_generation",
        base_input={
            "scientific_question": "苯甲醛红外与拉曼光谱",
            "task_config": {"num_queries": 2},
        },
        stages=stages,
        provider=provider,
        message_builder=_builder(calls),
        transport_attempts=transport_attempts,
    )
    return runner, provider_calls, calls


def _read_ref(task_root, ref):
    assert set(ref) == {"relative_path", "sha256", "media_type", "size"}
    path = task_root / ref["relative_path"]
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == ref["sha256"]
    assert len(payload) == ref["size"]
    return json.loads(payload)


class _RecordingModelStageObserver:
    """测试用观察器：保存事件深拷贝，并故意修改回调入参。"""

    def __init__(self):
        self.started = []
        self.attempts = []
        self.finished = []
        self.degraded = []
        self.first_stage_started = threading.Event()
        self.second_stage_started = threading.Event()

    def start_model_stage(self, event):
        self.started.append(copy.deepcopy(event))
        event["stage_input"]["observer_mutation"] = True
        contract_schema = event["response_contract"]["schema"]
        if contract_schema is not None:
            contract_schema["properties"]["value"]["type"] = "string"
        self.first_stage_started.set()
        if len(self.started) >= 2:
            self.second_stage_started.set()
        return {"stage_index": event["stage_index"]}

    def record_model_attempt(self, handle, event):
        self.attempts.append((copy.deepcopy(handle), copy.deepcopy(event)))
        event["compiled_request"]["observer_mutation"] = True

    def finish_model_stage(self, handle, event):
        self.finished.append((copy.deepcopy(handle), copy.deepcopy(event)))
        event["source_artifacts"]["observer_mutation"] = True

    def mark_model_output_degraded(self, reason):
        self.degraded.append(reason)


def test_model_stage_observer_records_success_without_changing_formal_manifest(
    tmp_path,
):
    """成功阶段应记录完整旁路事件，且终态引用已经冻结的 manifest。"""

    observer = _RecordingModelStageObserver()
    runner = model_operation.ModelOperationRunner(
        task_root=tmp_path,
        operation_instance_id="observer-success",
        operation="hypothesis_generation",
        base_input={"scientific_question": "观察成功阶段。"},
        stages=[_json_stage()],
        provider=lambda _request: _envelope('{"value":7}'),
        message_builder=_builder([]),
        transport_attempts=1,
        observer=observer,
        display_context={"phase": "hypothesis", "batch_index": 2},
    )

    result = runner.run_model_operation()

    assert len(observer.started) == 1
    assert len(observer.attempts) == 1
    assert len(observer.finished) == 1
    started = observer.started[0]
    _, attempt = observer.attempts[0]
    _, finished = observer.finished[0]
    assert started["operation"] == "hypothesis_generation"
    assert started["operation_id"] == "observer-success"
    assert started["raw_stage"] == "hypothesis"
    assert started["query_index"] == 1
    assert started["validation_kind"] == "json_schema"
    assert started["domain_validator_id"] is None
    assert started["response_contract"] == {
        "validation_kind": "json_schema",
        "schema": _json_stage().schema,
        "domain_validator_id": None,
    }
    assert started["display_context"] == {"phase": "hypothesis", "batch_index": 2}
    assert started["finished_at"] is None
    assert started["duration_ms"] is None
    assert attempt["compiled_request"]["catalog_key"] == (
        "hypothesis_generation.hypothesis"
    )
    assert attempt["response_contract"] == started["response_contract"]
    assert attempt["raw_response"]["id"] == "trace-1"
    assert attempt["parsed_output"] == {"value": 7}
    assert attempt["validation"]["status"] == "valid"
    assert attempt["source_artifact_refs"]["response_ref"] is not None
    assert finished["status"] == "validated"
    assert finished["final_output"] == {"value": 7}
    assert finished["started_at"].endswith("Z")
    assert finished["finished_at"].endswith("Z")
    assert finished["duration_ms"] >= 0
    assert finished["source_artifacts"]["manifest"] == result["manifest_ref"]
    assert finished["source_artifacts"]["base_input"] == result["manifest"][
        "base_input_ref"
    ]
    assert "observer_mutation" not in result["manifest"]["base_input_ref"]
    assert "observer_mutation" not in observer.attempts[0][1]["compiled_request"]
    assert "observer_mutation" not in finished["source_artifacts"]
    assert set(result["manifest"]) == {
        "schema_version",
        "operation_instance_id",
        "operation",
        "base_input_ref",
        "prior_manifest_ref",
        "stages",
    }


def test_model_stage_observer_keeps_null_response_schema_for_non_json_stage(tmp_path):
    """非 JSON 阶段的 response_contract 应保留 schema=null。"""

    observer = _RecordingModelStageObserver()
    relationship_stage = model_operation.ModelStage(
        stage="relationship",
        stage_kind="model",
        validation_kind="text_enum_relationship",
    )
    runner = model_operation.ModelOperationRunner(
        task_root=tmp_path,
        operation_instance_id="observer-contract-null",
        operation="hypothesis_generation",
        base_input={"scientific_question": "观察空 schema。"},
        stages=[relationship_stage],
        provider=lambda _request: _envelope("independent"),
        message_builder=_builder([]),
        transport_attempts=1,
        observer=observer,
    )

    result = runner.run_model_operation()

    assert result["status"] == "validated"
    assert observer.started[0]["response_contract"] == {
        "validation_kind": "text_enum_relationship",
        "schema": None,
        "domain_validator_id": None,
    }
    assert observer.attempts[0][1]["response_contract"] == observer.started[0][
        "response_contract"
    ]


def test_model_stage_observer_records_invalid_attempt_before_valid_retry(tmp_path):
    """协议错误重答应产生两个逻辑 attempt，而不是覆盖首个错误回答。"""

    observer = _RecordingModelStageObserver()
    runner, _, _ = _runner(
        tmp_path,
        [_envelope('{"value":"wrong"}'), _envelope('{"value":8}', trace_id="trace-2")],
        [_json_stage()],
        operation_instance_id="observer-retry",
    )
    runner.observer = observer
    runner.display_context = {"phase": "hypothesis"}

    result = runner.run_model_operation()

    assert result["status"] == "validated"
    assert [event[1]["validation"]["status"] for event in observer.attempts] == [
        "invalid",
        "valid",
    ]
    assert observer.attempts[0][1]["raw_response"]["choices"]
    assert observer.attempts[0][1]["parsed_output"] == {"value": "wrong"}
    assert observer.finished[0][1]["attempt_count"] == 2
    assert len(observer.finished[0][1]["source_artifacts"]["request_files"]) == 2
    assert len(observer.finished[0][1]["source_artifacts"]["response_files"]) == 2
    assert len(observer.finished[0][1]["source_artifacts"]["error_files"]) == 1
    assert len(result["manifest"]["stages"][0]["attempts"]) == 2


def test_model_stage_observer_records_one_transport_failure_after_network_retries(
    tmp_path,
):
    """三次底层传输重试仍只记录一个最终 transport_failed 逻辑回答。"""

    observer = _RecordingModelStageObserver()
    runner, _, _ = _runner(
        tmp_path,
        [ConnectionError("one"), TimeoutError("two"), RuntimeError("three")],
        [_json_stage()],
        operation_instance_id="observer-transport",
        transport_attempts=3,
    )
    runner.observer = observer

    result = runner.run_model_operation()

    assert result["status"] == "failed"
    assert len(observer.attempts) == 1
    attempt = observer.attempts[0][1]
    assert attempt["raw_response"] is None
    assert attempt["parsed_output"] is None
    assert attempt["validation"]["status"] == "transport_failed"
    assert attempt["validation"]["summary"] == (
        "provider transport attempts were exhausted"
    )
    assert observer.finished[0][1]["status"] == "failed"
    assert observer.finished[0][1]["source_artifacts"]["manifest"] == result[
        "manifest_ref"
    ]


def test_model_stage_observer_failure_cannot_change_formal_result(tmp_path):
    """观察器抛错时，正式模型结果、清单和 provider 调用仍保持不变。"""

    class BrokenObserver:
        def start_model_stage(self, _event):
            raise RuntimeError("observer failure")

        def record_model_attempt(self, _handle, _event):
            raise AssertionError("不会再次调用已隔离的观察器")

        def finish_model_stage(self, _handle, _event):
            raise AssertionError("不会再次调用已隔离的观察器")

    provider_calls = []
    runner = model_operation.ModelOperationRunner(
        task_root=tmp_path,
        operation_instance_id="observer-error",
        operation="hypothesis_generation",
        base_input={"scientific_question": "观察器异常隔离。"},
        stages=[_json_stage()],
        provider=lambda request: (
            provider_calls.append(request) or _envelope('{"value":9}')
        ),
        message_builder=_builder([]),
        transport_attempts=1,
        observer=BrokenObserver(),
    )

    result = runner.run_model_operation()

    assert result["status"] == "validated"
    assert result["values"] == [{"value": 9}]
    assert len(provider_calls) == 1
    assert set(result["manifest"]) == {
        "schema_version",
        "operation_instance_id",
        "operation",
        "base_input_ref",
        "prior_manifest_ref",
        "stages",
    }


def test_dynamic_observer_records_real_parallel_stage_start_order(tmp_path):
    """动态并行 stage 按真实启动顺序发开始事件，而不是按完成顺序重排。"""

    observer = _RecordingModelStageObserver()
    first_provider_started = threading.Event()
    release_first_provider = threading.Event()
    provider_calls = []

    def provider(_request):
        provider_calls.append(True)
        if len(provider_calls) == 1:
            first_provider_started.set()
            assert release_first_provider.wait(timeout=5)
        return _envelope('{"value":10}')

    runner = model_operation.DynamicModelOperationRunner(
        task_root=tmp_path,
        operation_instance_id="observer-dynamic-order",
        operation="hypothesis_generation",
        base_input={"scientific_question": "动态观察顺序。"},
        provider=provider,
        message_builder=_builder([]),
        transport_attempts=1,
        observer=observer,
    )
    stage_one = _json_stage()
    stage_two = _json_stage()
    slots = runner.reserve_model_stages([stage_one, stage_two])
    errors = []

    def call_stage(slot):
        try:
            runner.run_reserved_model_stage(slot, stage_one, {"slot": slot})
        except BaseException as exc:  # pragma: no cover - 仅回传线程异常
            errors.append(exc)

    first = threading.Thread(target=call_stage, args=(slots[0],), daemon=True)
    second = threading.Thread(target=call_stage, args=(slots[1],), daemon=True)
    first.start()
    assert first_provider_started.wait(timeout=5)
    second.start()
    assert observer.second_stage_started.wait(timeout=5)
    release_first_provider.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not errors
    assert [event["stage_index"] for event in observer.started] == [1, 2]
    assert runner.finalize_model_operation()["status"] == "validated"


def test_dynamic_observer_drops_base_exception_attempt_when_retry_succeeds(
    tmp_path,
):
    """同一动态 stage 的瞬时异常重试成功时，只发送最终一个 validated 终态。"""

    observer = _RecordingModelStageObserver()
    responses = [KeyboardInterrupt("瞬时中断"), _envelope('{"value":11}')]

    def provider(_request):
        response = responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    runner = model_operation.DynamicModelOperationRunner(
        task_root=tmp_path,
        operation_instance_id="observer-base-exception-retry",
        operation="hypothesis_generation",
        base_input={"scientific_question": "观察异常后重试。"},
        provider=provider,
        message_builder=_builder([]),
        transport_attempts=1,
        observer=observer,
    )
    stage = _json_stage()
    slot = runner.reserve_model_stages([stage])[0]

    with pytest.raises(KeyboardInterrupt, match="瞬时中断"):
        runner.run_reserved_model_stage(slot, stage, {"query_angle": "重试"})

    assert observer.finished == []
    assert runner.run_reserved_model_stage(
        slot,
        stage,
        {"query_angle": "重试"},
    ) == {"value": 11}

    result = runner.finalize_model_operation()

    assert result["status"] == "validated"
    assert len(observer.started) == 2
    assert len(observer.attempts) == 1
    assert len(observer.finished) == 1
    finished = observer.finished[0][1]
    assert finished["status"] == "validated"
    assert finished["final_output"] == {"value": 11}
    assert finished["attempt_count"] == 1


def test_dynamic_observer_records_valid_attempt_after_staging_retry(
    tmp_path,
    monkeypatch,
):
    """有效回答暂存失败后，终态事件仍包含正式采用的回答和引用。"""

    observer = _RecordingModelStageObserver()
    runner = model_operation.DynamicModelOperationRunner(
        task_root=tmp_path,
        operation_instance_id="observer-staging-retry",
        operation="hypothesis_generation",
        base_input={"scientific_question": "观察有效回答暂存失败。"},
        provider=lambda _request: _envelope('{"value":12}'),
        message_builder=_builder([]),
        transport_attempts=1,
        observer=observer,
    )
    stage = _json_stage()
    slot = runner.reserve_model_stages([stage])[0]
    original_write_staging = runner._write_staging
    remaining_failures = [True]

    def fail_once(manifest):
        if remaining_failures:
            remaining_failures.pop()
            raise OSError("注入有效回答暂存失败")
        return original_write_staging(manifest)

    monkeypatch.setattr(runner, "_write_staging", fail_once)
    with pytest.raises(OSError, match="有效回答暂存失败"):
        runner.run_reserved_model_stage(
            slot,
            stage,
            {"query_angle": "暂存失败"},
        )

    assert len(observer.attempts) == 1
    monkeypatch.setattr(runner, "_write_staging", original_write_staging)
    result = runner.finalize_model_operation()

    assert result["status"] == "validated"
    assert len(result["manifest"]["stages"][0]["attempts"]) == 1
    assert len(observer.finished) == 1
    finished = observer.finished[0][1]
    assert finished["status"] == "validated"
    assert finished["final_output"] == {"value": 12}
    assert finished["attempt_count"] == 1
    assert len(finished["source_artifacts"]["attempts"]) == 1
    assert finished["source_artifacts"]["attempts"][0]["response_ref"] is not None


def test_recovered_validated_prefix_does_not_emit_fake_observer_call(tmp_path):
    """恢复旧成功前缀只重放 artifact，不产生新的开始或 attempt 事件。"""

    stage = _json_stage()
    first_observer = _RecordingModelStageObserver()
    responses = [_envelope('{"value":1}')] + [_envelope("invalid") for _ in range(4)]
    first = model_operation.DynamicModelOperationRunner(
        task_root=tmp_path,
        operation_instance_id="observer-prefix-first",
        operation="hypothesis_generation",
        base_input={"scientific_question": "恢复观察。"},
        provider=lambda _request: responses.pop(0),
        message_builder=_builder([]),
        transport_attempts=1,
        observer=first_observer,
    )
    assert first.run_model_stage(stage, {"slot": 1}) == {"value": 1}
    with pytest.raises(model_operation.ModelOperationStageError):
        first.run_model_stage(stage, {"slot": 2})
    failed = first.finalize_model_operation()
    assert [event["stage_index"] for event in first_observer.started] == [1, 2]

    retry_observer = _RecordingModelStageObserver()
    retry = model_operation.DynamicModelOperationRunner.from_failed_manifest(
        task_root=tmp_path,
        operation_instance_id="observer-prefix-retry",
        prior_manifest_path=failed["manifest_path"],
        provider=lambda _request: _envelope('{"value":2}'),
        message_builder=_builder([]),
        transport_attempts=1,
        observer=retry_observer,
    )
    assert retry.run_model_stage(stage, {"slot": 1}) == {"value": 1}
    assert retry.run_model_stage(stage, {"slot": 2}) == {"value": 2}
    recovered = retry.finalize_model_operation()

    assert recovered["status"] == "validated"
    assert [event["stage_index"] for event in retry_observer.started] == [2]
    assert [event["stage_index"] for _, event in retry_observer.attempts] == [2]


def test_finalized_model_operation_can_be_replayed_with_exact_identity(tmp_path):
    """提交方必须能从最终清单复验输入、响应和通过领域校验的值。"""

    stage = model_operation.ModelStage(
        stage="router",
        stage_kind="model",
        validation_kind="json_schema",
        schema={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "properties": {"decision": {"const": "evidence_ready"}},
            "required": ["decision"],
        },
        domain_validator=lambda value: {
            "decision": value["decision"],
            "validated": True,
        },
        domain_validator_id="test.router.replay.v1",
    )
    calls = []
    runner = model_operation.ModelOperationRunner(
        task_root=tmp_path,
        operation_instance_id="router-replay",
        operation="router",
        base_input={"scientific_question": "验证最终清单重放。"},
        stages=[stage],
        provider=lambda _messages: _envelope(
            json.dumps({"decision": "evidence_ready"})
        ),
        message_builder=_builder(calls),
    )
    result = runner.run_model_operation()

    replayed = model_operation.ModelOperationRunner.replay_manifest(
        task_root=tmp_path,
        manifest_path=result["manifest_path"],
        operation="router",
        operation_instance_id="router-replay",
        expected_base_input={"scientific_question": "验证最终清单重放。"},
        stages=[stage],
    )

    assert replayed["values"] == [
        {"decision": "evidence_ready", "validated": True}
    ]
    assert replayed["manifest"] == result["manifest"]


def test_replay_rejects_a_changed_domain_validator_identity(tmp_path):
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
    }
    original_stage = model_operation.ModelStage(
        stage="router",
        stage_kind="model",
        validation_kind="json_schema",
        schema=schema,
        domain_validator=lambda value: value,
        domain_validator_id="test.router.validator.v1",
    )
    runner = model_operation.ModelOperationRunner(
        task_root=tmp_path,
        operation_instance_id="validator-identity",
        operation="router",
        base_input={"scientific_question": "验证校验器身份。"},
        stages=[original_stage],
        provider=lambda _request: _envelope('{"value":1}'),
        message_builder=_builder([]),
    )
    result = runner.run_model_operation()
    changed_stage = model_operation.ModelStage(
        stage="router",
        stage_kind="model",
        validation_kind="json_schema",
        schema=schema,
        domain_validator=lambda value: value,
        domain_validator_id="test.router.validator.v2",
    )

    with pytest.raises(
        model_operation.ModelOperationError,
        match="compiled request",
    ):
        model_operation.ModelOperationRunner.replay_manifest(
            task_root=tmp_path,
            manifest_path=result["manifest_path"],
            operation="router",
            operation_instance_id="validator-identity",
            expected_base_input={"scientific_question": "验证校验器身份。"},
            stages=[changed_stage],
        )


def test_all_four_model_validation_kinds_use_their_registered_protocol(tmp_path):
    stages = [
        model_operation.ModelStage(
            stage="query",
            stage_kind="model",
            validation_kind="query_delimited_text",
            expected_count=2,
        ),
        _json_stage(domain_validator=lambda value: {"normalized": value["value"]}),
        model_operation.ModelStage(
            stage="relationship",
            stage_kind="model",
            validation_kind="text_enum_relationship",
        ),
        model_operation.ModelStage(
            stage="rank",
            stage_kind="model",
            validation_kind="text_enum_rank",
        ),
    ]
    runner, provider_calls, _ = _runner(
        tmp_path,
        [
            _envelope("角度一<>角度二", trace_id="trace-query"),
            _envelope('{"value":2}', trace_id="trace-json"),
            _envelope(" Merge ", trace_id="trace-relationship"),
            _envelope(" b ", trace_id="trace-rank"),
        ],
        stages,
    )

    result = runner.run_model_operation()

    assert result["status"] == "validated"
    assert result["values"] == [
        ["角度一", "角度二"],
        {"normalized": 2},
        "merge",
        "B",
    ]
    assert len(provider_calls) == 4
    assert [item["status"] for item in result["manifest"]["stages"]] == [
        "validated",
        "validated",
        "validated",
        "validated",
    ]


@pytest.mark.parametrize(
    ("response", "failure_kind", "finish_reason"),
    [
        ({"id": "trace", "usage": {}}, "provider_envelope", None),
        ({"id": "trace", "choices": {}, "usage": {}}, "provider_envelope", None),
        ({"id": "trace", "choices": [], "usage": {}}, "empty_choices", None),
        (_envelope("not-json", finish_reason="length"), "truncated", "length"),
        (_envelope("", finish_reason="max_tokens"), "truncated", "max_tokens"),
        (_envelope("   "), "empty_content", "stop"),
        (_envelope("说明\n```json\n{\"value\":1}\n```"), "json_syntax", "stop"),
        (_envelope('{"value":NaN}'), "json_syntax", "stop"),
        (_envelope('{"value":1,"value":2}'), "json_syntax", "stop"),
        (_envelope('{"value":"\\ud800"}'), "json_syntax", "stop"),
        (_envelope('\u00a0{"value":1}'), "json_syntax", "stop"),
        (_envelope('{"value":1}', finish_reason="content_filter"), "provider_envelope", "content_filter"),
        (_envelope('{"value":"wrong"}'), "schema", "stop"),
        (_envelope('{"value":0}'), "domain", "stop"),
    ],
)
def test_invalid_provider_and_protocol_responses_are_classified_in_priority_order(
    tmp_path, response, failure_kind, finish_reason
):
    domain_calls = []

    def domain_validator(value):
        domain_calls.append(value)
        if value["value"] <= 0:
            raise ValueError("领域值无效且不得写入错误 artifact")
        return value

    runner, provider_calls, _ = _runner(
        tmp_path,
        [response, response, response, response],
        [_json_stage(domain_validator=domain_validator)],
    )

    result = runner.run_model_operation()

    assert result["status"] == "failed"
    assert result["failure_kind"] == failure_kind
    assert len(provider_calls) == 4
    attempts = result["manifest"]["stages"][0]["attempts"]
    assert len(attempts) == 4
    for attempt in attempts:
        assert attempt["validation_status"] == "invalid"
        error = _read_ref(tmp_path, attempt["error_ref"])
        assert error["failure_kind"] == failure_kind
        assert error["finish_reason"] == finish_reason
        assert set(error) == {"failure_kind", "finish_reason", "summary"}
        if failure_kind == "domain":
            assert "领域值无效" in error["summary"]
            assert error["summary"].startswith("domain_violation.v1:")
        else:
            assert "领域值无效" not in error["summary"]
        assert "/tmp" not in error["summary"]
    if failure_kind in {
        "provider_envelope",
        "empty_choices",
        "truncated",
        "empty_content",
        "json_syntax",
        "schema",
    }:
        assert domain_calls == []
    else:
        assert len(domain_calls) == 4


def test_complete_markdown_json_fence_is_unwrapped_without_retry(tmp_path):
    """只有完整包住单个 JSON 值的围栏可以去壳，原始响应仍由审计文件保存。"""

    runner, provider_calls, _ = _runner(
        tmp_path,
        [_envelope("```json\n{\"value\":2}\n```")],
        [_json_stage()],
    )

    result = runner.run_model_operation()

    assert result["status"] == "validated"
    assert result["values"] == [{"value": 2}]
    assert len(provider_calls) == 1
    attempt = result["manifest"]["stages"][0]["attempts"][0]
    raw_response = _read_ref(tmp_path, attempt["response_ref"])
    assert raw_response["choices"][0]["message"]["content"].startswith(
        "```json"
    )


def test_fenced_json_keeps_duplicate_field_rejection():
    """去掉外层围栏后仍不能静默覆盖内容冲突的重复字段。"""

    result = model_operation.validate_model_envelope(
        _envelope("```json\n{\"value\":1,\"value\":2}\n```"),
        _json_stage(),
    )

    assert result.failure_kind == "json_syntax"
    assert "duplicate object field" in result.summary


def test_identical_duplicate_field_is_merged_without_retry(tmp_path):
    """完全相同的重复字段不改变含义，不应触发一次昂贵的模型重试。"""

    runner, provider_calls, _ = _runner(
        tmp_path,
        [_envelope('{"value":2,"value":2}')],
        [_json_stage()],
    )

    result = runner.run_model_operation()

    assert result["status"] == "validated"
    assert result["values"] == [{"value": 2}]
    assert len(provider_calls) == 1
    attempt = result["manifest"]["stages"][0]["attempts"][0]
    raw_response = _read_ref(tmp_path, attempt["response_ref"])
    assert raw_response["choices"][0]["message"]["content"] == (
        '{"value":2,"value":2}'
    )


def test_protocol_regeneration_uses_only_safe_error_and_same_base_input(tmp_path):
    calls = []

    def domain_validator(value):
        if value["value"] <= 0:
            raise ValueError("value 必须大于 0；当前值不满足领域规则")
        return value

    runner, provider_calls, builder_calls = _runner(
        tmp_path,
        [
            _envelope('{"value":1,"value":2}'),
            _envelope('{"value":"wrong"}'),
            _envelope('{"value":0}'),
            _envelope('{"value":2}'),
        ],
        [_json_stage(domain_validator=domain_validator)],
        call_log=calls,
    )

    result = runner.run_model_operation()

    assert result["status"] == "validated"
    assert result["values"] == [{"value": 2}]
    assert len(provider_calls) == 4
    assert [item["protocol_error"] for item in builder_calls] == [
        None,
        {
            "failure_kind": "json_syntax",
            "summary": 'json_syntax.v1:{"column":0,"line":0,"reason":"duplicate object field"}',
        },
        {
            "failure_kind": "schema",
            "summary": (
                'schema_violation.v1:[{"expected":"integer","observed":"string",'
                '"path":"/value","rule":"type"}]; earlier stage failures must '
                'not recur: json_syntax: json_syntax.v1:{"column":0,"line":0,'
                '"reason":"duplicate object field"}'
            ),
        },
        {
            "failure_kind": "domain",
            "summary": (
                'domain_violation.v1:{"reason":"value 必须大于 0；当前值不满足领域规则",'
                '"validator":"test.integer.v1"}; earlier '
                'stage failures must not recur: json_syntax: json_syntax.v1:'
                '{"column":0,"line":0,"reason":"duplicate object field"} | '
                'schema: schema_violation.v1:[{"expected":'
                '"integer","observed":"string","path":"/value","rule":"type"}]'
            ),
        },
    ]
    assert all(
        item["base_input"] == builder_calls[0]["base_input"]
        for item in builder_calls
    )
    assert "```json" not in json.dumps(builder_calls, ensure_ascii=False)


def test_schema_regeneration_reports_all_actionable_coordinates(tmp_path):
    """缺字段、错类型、非法枚举和多余字段都必须告诉模型具体修正位置。"""

    stage = model_operation.ModelStage(
        stage="hypothesis",
        stage_kind="model",
        validation_kind="json_schema",
        query_index=1,
        schema={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "value": {"type": "integer"},
                "mode": {"enum": ["A", "B"]},
                "required_note": {"type": "string"},
            },
            "required": ["value", "mode", "required_note"],
        },
    )
    runner, _, builder_calls = _runner(
        tmp_path,
        [
            _envelope(
                json.dumps(
                    {
                        "value": f"SECRET {_ABSOLUTE_PRIVATE_EVIDENCE}",
                        "mode": "wrong",
                        "unexpected": "SECRET OLD RESPONSE",
                    }
                )
            ),
            _envelope('{"mode":"A","required_note":"ok","value":1}'),
        ],
        [stage],
    )

    result = runner.run_model_operation()

    assert result["status"] == "validated"
    summary = builder_calls[1]["protocol_error"]["summary"]
    assert summary.startswith("schema_violation.v1:")
    details = json.loads(summary.split(":", 1)[1])
    assert {
        (item["path"], item["rule"])
        for item in details
    } == {
        ("/", "additionalProperties"),
        ("/", "required"),
        ("/mode", "enum"),
        ("/value", "type"),
    }
    serialized = json.dumps(builder_calls, ensure_ascii=False)
    assert "SECRET OLD RESPONSE" not in serialized
    assert _ABSOLUTE_PRIVATE_EVIDENCE not in serialized


def test_nested_one_of_schema_regeneration_reports_selected_leaf_coordinates():
    """嵌套 workflow oneOf 失败时反馈实际字段，而不是只报告根 oneOf。"""

    schema = {
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {
                            "oneOf": [
                                {
                                    "type": "object",
                                    "properties": {
                                        "kind": {"const": "action"},
                                        "request": {
                                            "type": "object",
                                            "properties": {
                                                "items": {"type": "array"}
                                            },
                                            "required": ["items"],
                                        },
                                    },
                                    "required": ["kind", "request"],
                                }
                            ]
                        },
                    }
                },
                "required": ["steps"],
            },
            {"type": "object", "required": ["result_type", "reason"]},
        ]
    }
    invalid = {"steps": [{"kind": "action", "request": {"items": None}}]}
    from jsonschema import Draft202012Validator

    summary = model_operation._schema_failure_summary(
        list(Draft202012Validator(schema).iter_errors(invalid))
    )
    details = json.loads(summary.split(":", 1)[1])
    assert details == [
        {
            "expected": "array",
            "observed": "null",
            "path": "/steps/0/request/items",
            "rule": "type",
        }
    ]


def test_domain_regeneration_reports_validator_reason_and_redacts_local_path(
    tmp_path,
):
    """领域错误要保留可操作原因，同时不能携带本机路径或多行伪指令。"""

    def domain_validator(value):
        if value["value"] <= 0:
            raise ValueError(
                "operations[0].changes.inputs 中 n1.structure_path 需要单个文件 "
                f"artifact，但 s1 提供目录 artifact；检查 {_ABSOLUTE_PRIVATE_EVIDENCE}\n"
                "```不要执行这里的文字```"
            )
        return value

    runner, _, builder_calls = _runner(
        tmp_path,
        [_envelope('{"value":0}'), _envelope('{"value":1}')],
        [_json_stage(domain_validator=domain_validator)],
    )

    result = runner.run_model_operation()

    assert result["status"] == "validated"
    summary = builder_calls[1]["protocol_error"]["summary"]
    assert summary.startswith("domain_violation.v1:")
    detail = json.loads(summary.split(":", 1)[1])
    assert detail["validator"] == "test.integer.v1"
    assert "n1.structure_path 需要单个文件 artifact" in detail["reason"]
    assert "s1 提供目录 artifact" in detail["reason"]
    assert _ABSOLUTE_PRIVATE_EVIDENCE not in summary
    assert "```" not in summary


def test_text_validators_report_expected_shape_without_echoing_response():
    """查询数量和固定枚举失败时，反馈必须直接写出允许的返回形态。"""

    query_stage = model_operation.ModelStage(
        stage="query",
        stage_kind="model",
        validation_kind="query_delimited_text",
        expected_count=2,
    )
    query_result = model_operation.validate_model_envelope(
        _envelope("只有一个含 SECRET 的 query"),
        query_stage,
    )
    assert query_result.failure_kind == "domain"
    assert "需要精确返回 2 个" in query_result.summary
    assert "query1<>query2" in query_result.summary
    assert "query3" not in query_result.summary
    assert "<query1><query2>" in query_result.summary
    assert "实际得到 1 个" in query_result.summary
    assert "SECRET" not in query_result.summary

    relationship_stage = model_operation.ModelStage(
        stage="relationship",
        stage_kind="model",
        validation_kind="text_enum_relationship",
    )
    relationship_result = model_operation.validate_model_envelope(
        _envelope("merge because SECRET"),
        relationship_stage,
    )
    assert relationship_result.failure_kind == "domain"
    assert "merge、complement、conflict 或 independent" in relationship_result.summary
    assert "SECRET" not in relationship_result.summary


def test_non_fenced_json_error_does_not_claim_markdown_was_observed(tmp_path):
    """普通语法错误不能被误报成 Markdown 围栏，避免给模型错误反馈。"""

    runner, _, builder_calls = _runner(
        tmp_path,
        [_envelope("not-json"), _envelope('{"value":2}')],
        [_json_stage()],
    )

    result = runner.run_model_operation()

    assert result["status"] == "validated"
    assert builder_calls[1]["protocol_error"] == {
        "failure_kind": "json_syntax",
        "summary": (
            'json_syntax.v1:{"column":1,"line":1,"reason":"syntax error"}'
        ),
    }


def test_fixed_and_dynamic_model_stage_protocols_match(tmp_path):
    """固定与动态 runner 对同一 stage 使用相同的请求和 attempt 协议。"""

    stage_input = {
        "scientific_question": "比较固定与动态模型阶段。",
        "request_context": {"scope": "same-stage"},
    }
    responses = [_envelope('{"value":"wrong"}'), _envelope('{"value":8}')]

    def run_provider(queue):
        def provider(_request):
            return queue.pop(0)

        return provider

    fixed = model_operation.ModelOperationRunner(
        task_root=tmp_path / "fixed",
        operation_instance_id="fixed-stage-protocol",
        operation="hypothesis_generation",
        base_input=stage_input,
        stages=[_json_stage()],
        provider=run_provider(list(responses)),
        message_builder=_builder([]),
        transport_attempts=1,
    )
    fixed_result = fixed.run_model_operation()

    dynamic = model_operation.DynamicModelOperationRunner(
        task_root=tmp_path / "dynamic",
        operation_instance_id="dynamic-stage-protocol",
        operation="hypothesis_generation",
        base_input={"scientific_question": "动态操作外层输入。"},
        provider=run_provider(list(responses)),
        message_builder=_builder([]),
        transport_attempts=1,
    )
    dynamic_stage = _json_stage()
    dynamic_slot = dynamic.reserve_model_stages([dynamic_stage])[0]
    dynamic_value = dynamic.run_reserved_model_stage(
        dynamic_slot,
        dynamic_stage,
        stage_input,
    )
    dynamic_result = dynamic.finalize_model_operation()

    def read_attempt(root, attempt):
        def read_ref(ref):
            return None if ref is None else _read_ref(root, ref)

        return {
            "attempt_index": attempt["attempt_index"],
            "request": read_ref(attempt["request_ref"]),
            "response": read_ref(attempt["response_ref"]),
            "provider_trace_id": attempt["provider_trace_id"],
            "error": read_ref(attempt["error_ref"]),
            "validation_status": attempt["validation_status"],
        }

    fixed_attempts = fixed_result["manifest"]["stages"][0]["attempts"]
    dynamic_attempts = dynamic_result["manifest"]["stages"][0]["attempts"]
    assert fixed_result["values"] == [{"value": 8}]
    assert dynamic_value == {"value": 8}
    assert dynamic_result["status"] == "validated"
    assert [
        read_attempt(tmp_path / "fixed", attempt) for attempt in fixed_attempts
    ] == [
        read_attempt(tmp_path / "dynamic", attempt)
        for attempt in dynamic_attempts
    ]


def test_dynamic_same_slot_is_occupied_until_provider_returns(tmp_path):
    """同一动态 slot 在首个 provider 返回前不能启动第二次调用。"""

    provider_started = threading.Event()
    release_provider = threading.Event()
    provider_calls = []

    def provider(request):
        provider_calls.append(request)
        if len(provider_calls) > 1:
            raise AssertionError("同一 slot 不应第二次进入 provider")
        provider_started.set()
        assert release_provider.wait(timeout=5)
        return _envelope('{"value":11}')

    runner = model_operation.DynamicModelOperationRunner(
        task_root=tmp_path,
        operation_instance_id="dynamic-active-slot",
        operation="hypothesis_generation",
        base_input={"scientific_question": "验证动态 slot 占用。"},
        provider=provider,
        message_builder=_builder([]),
        transport_attempts=1,
    )
    stage = _json_stage()
    slot = runner.reserve_model_stages([stage])[0]
    first_result = {}

    def run_first():
        try:
            first_result["value"] = runner.run_reserved_model_stage(
                slot,
                stage,
                {"query_angle": "并发占用"},
            )
        except BaseException as exc:  # pragma: no cover - 仅用于向主线程传递异常
            first_result["error"] = exc

    first_thread = threading.Thread(target=run_first, daemon=True)
    first_thread.start()
    try:
        assert provider_started.wait(timeout=5)
        with pytest.raises(model_operation.ModelOperationError, match="正在执行"):
            runner.run_reserved_model_stage(
                slot,
                stage,
                {"query_angle": "并发占用"},
            )
        assert len(provider_calls) == 1
    finally:
        release_provider.set()
        first_thread.join(timeout=5)

    assert not first_thread.is_alive()
    assert "error" not in first_result
    assert first_result["value"] == {"value": 11}
    finalized = runner.finalize_model_operation()
    assert finalized["status"] == "validated"
    assert len(finalized["manifest"]["stages"][0]["attempts"]) == 1


def test_dynamic_slot_clears_active_after_message_builder_error(tmp_path):
    """消息构造意外失败后，slot 可以再次显式调用且不会伪造完成状态。"""

    builder_calls = []
    fallback_builder = _builder([])

    def builder(*args, **kwargs):
        builder_calls.append(True)
        if len(builder_calls) == 1:
            raise model_operation.ModelOperationError("注入消息构造失败")
        return fallback_builder(*args, **kwargs)

    runner = model_operation.DynamicModelOperationRunner(
        task_root=tmp_path,
        operation_instance_id="dynamic-active-cleanup",
        operation="hypothesis_generation",
        base_input={"scientific_question": "验证异常清理。"},
        provider=lambda _request: _envelope('{"value":12}'),
        message_builder=builder,
        transport_attempts=1,
    )
    stage = _json_stage()
    slot = runner.reserve_model_stages([stage])[0]

    with pytest.raises(model_operation.ModelOperationError, match="消息构造失败"):
        runner.run_reserved_model_stage(slot, stage, {"query_angle": "异常"})
    assert runner.run_reserved_model_stage(
        slot,
        stage,
        {"query_angle": "异常"},
    ) == {"value": 12}
    assert runner.finalize_model_operation()["status"] == "validated"


def test_dynamic_valid_attempt_staging_failure_keeps_completed_slot(
    tmp_path,
    monkeypatch,
):
    """有效回答已进入内存清单后，staging 写盘失败不能重复调用 provider。"""

    provider_calls = []

    def provider(request):
        provider_calls.append(request)
        return _envelope('{"value":13}')

    runner = model_operation.DynamicModelOperationRunner(
        task_root=tmp_path,
        operation_instance_id="dynamic-staging-failure",
        operation="hypothesis_generation",
        base_input={"scientific_question": "验证 staging 写盘失败。"},
        provider=provider,
        message_builder=_builder([]),
        transport_attempts=1,
    )
    stage = _json_stage()
    slot = runner.reserve_model_stages([stage])[0]
    original_write_staging = runner._write_staging
    remaining_faults = [1]

    def fail_once(manifest):
        if remaining_faults:
            remaining_faults.pop()
            raise OSError("注入 staging 写盘失败")
        return original_write_staging(manifest)

    monkeypatch.setattr(runner, "_write_staging", fail_once)
    with pytest.raises(OSError, match="staging 写盘失败"):
        runner.run_reserved_model_stage(slot, stage, {"query_angle": "写盘"})
    assert len(provider_calls) == 1

    monkeypatch.setattr(runner, "_write_staging", original_write_staging)
    with pytest.raises(model_operation.ModelOperationError, match="不能执行两次"):
        runner.run_reserved_model_stage(slot, stage, {"query_angle": "写盘"})
    finalized = runner.finalize_model_operation()
    assert finalized["status"] == "validated"
    attempts = finalized["manifest"]["stages"][0]["attempts"]
    assert len(attempts) == 1
    assert attempts[0]["validation_status"] == "valid"


def test_dynamic_invalid_staging_failure_rolls_back_attempt_before_retry(
    tmp_path,
    monkeypatch,
):
    """无效回答的 staging 写盘失败后，重试不应留下重复 attempt。"""

    provider_calls = []
    responses = [_envelope("not-json"), _envelope('{"value":14}')]

    def provider(request):
        provider_calls.append(request)
        return responses.pop(0)

    runner = model_operation.DynamicModelOperationRunner(
        task_root=tmp_path,
        operation_instance_id="dynamic-invalid-staging-failure",
        operation="hypothesis_generation",
        base_input={"scientific_question": "验证无效 staging 回滚。"},
        provider=provider,
        message_builder=_builder([]),
        transport_attempts=1,
    )
    stage = _json_stage()
    slot = runner.reserve_model_stages([stage])[0]
    original_write_staging = runner._write_staging
    remaining_faults = [1]

    def fail_once(manifest):
        if remaining_faults:
            remaining_faults.pop()
            raise OSError("注入 invalid staging 写盘失败")
        return original_write_staging(manifest)

    monkeypatch.setattr(runner, "_write_staging", fail_once)
    with pytest.raises(OSError, match="invalid staging 写盘失败"):
        runner.run_reserved_model_stage(slot, stage, {"query_angle": "无效"})
    assert len(provider_calls) == 1

    monkeypatch.setattr(runner, "_write_staging", original_write_staging)
    assert runner.run_reserved_model_stage(
        slot,
        stage,
        {"query_angle": "无效"},
    ) == {"value": 14}
    finalized = runner.finalize_model_operation()
    assert finalized["status"] == "validated"
    assert len(provider_calls) == 2
    attempts = finalized["manifest"]["stages"][0]["attempts"]
    assert [attempt["attempt_index"] for attempt in attempts] == [0]
    assert [attempt["validation_status"] for attempt in attempts] == ["valid"]


def test_dynamic_transport_staging_failure_rolls_back_attempt_before_retry(
    tmp_path,
    monkeypatch,
):
    """传输失败的 staging 写盘失败后，重试仍从唯一的零号 attempt 开始。"""

    provider_calls = []
    responses = [ConnectionError("受控传输失败"), _envelope('{"value":15}')]

    def provider(request):
        provider_calls.append(request)
        response = responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    runner = model_operation.DynamicModelOperationRunner(
        task_root=tmp_path,
        operation_instance_id="dynamic-transport-staging-failure",
        operation="hypothesis_generation",
        base_input={"scientific_question": "验证传输 staging 回滚。"},
        provider=provider,
        message_builder=_builder([]),
        transport_attempts=1,
    )
    stage = _json_stage()
    slot = runner.reserve_model_stages([stage])[0]
    original_write_staging = runner._write_staging
    remaining_faults = [1]

    def fail_once(manifest):
        if remaining_faults:
            remaining_faults.pop()
            raise OSError("注入 transport staging 写盘失败")
        return original_write_staging(manifest)

    monkeypatch.setattr(runner, "_write_staging", fail_once)
    with pytest.raises(OSError, match="transport staging 写盘失败"):
        runner.run_reserved_model_stage(slot, stage, {"query_angle": "传输"})
    assert len(provider_calls) == 1

    monkeypatch.setattr(runner, "_write_staging", original_write_staging)
    assert runner.run_reserved_model_stage(
        slot,
        stage,
        {"query_angle": "传输"},
    ) == {"value": 15}
    finalized = runner.finalize_model_operation()
    assert finalized["status"] == "validated"
    assert len(provider_calls) == 2
    attempts = finalized["manifest"]["stages"][0]["attempts"]
    assert [attempt["attempt_index"] for attempt in attempts] == [0]
    assert [attempt["validation_status"] for attempt in attempts] == ["valid"]


def test_dynamic_invalid_prefix_survives_staging_failure(
    tmp_path,
    monkeypatch,
):
    """后一次 invalid 写盘失败时，只撤销这一次并从旧错误继续重答。"""

    provider_calls = []
    builder_calls = []
    responses = [
        _envelope("not-json"),
        _envelope('{"value":"wrong"}'),
        _envelope('{"value":16}'),
    ]

    def provider(request):
        provider_calls.append(request)
        return responses.pop(0)

    runner = model_operation.DynamicModelOperationRunner(
        task_root=tmp_path,
        operation_instance_id="dynamic-invalid-prefix",
        operation="hypothesis_generation",
        base_input={"scientific_question": "验证 invalid 前缀恢复。"},
        provider=provider,
        message_builder=_builder(builder_calls),
        transport_attempts=1,
    )
    stage = _json_stage()
    slot = runner.reserve_model_stages([stage])[0]
    original_write_staging = runner._write_staging
    write_calls = [0]

    def fail_on_second_write(manifest):
        write_calls[0] += 1
        if write_calls[0] == 2:
            raise OSError("注入第二个 invalid staging 写盘失败")
        return original_write_staging(manifest)

    monkeypatch.setattr(runner, "_write_staging", fail_on_second_write)
    with pytest.raises(OSError, match="第二个 invalid staging 写盘失败"):
        runner.run_reserved_model_stage(slot, stage, {"query_angle": "恢复"})
    assert len(provider_calls) == 2

    monkeypatch.setattr(runner, "_write_staging", original_write_staging)
    assert runner.run_reserved_model_stage(
        slot,
        stage,
        {"query_angle": "恢复"},
    ) == {"value": 16}
    finalized = runner.finalize_model_operation()

    assert finalized["status"] == "validated"
    assert len(provider_calls) == 3
    attempts = finalized["manifest"]["stages"][0]["attempts"]
    assert [attempt["attempt_index"] for attempt in attempts] == [0, 1]
    assert [attempt["validation_status"] for attempt in attempts] == [
        "invalid",
        "valid",
    ]
    assert builder_calls[2]["protocol_error"] == {
        "failure_kind": "json_syntax",
        "summary": (
            'json_syntax.v1:{"column":1,"line":1,"reason":"syntax error"}'
        ),
    }


def test_dynamic_valid_after_invalid_keeps_history(
    tmp_path,
    monkeypatch,
):
    """有效回答写盘失败时，之前的 invalid 和有效回答都不能丢失。"""

    provider_calls = []
    responses = [_envelope("not-json"), _envelope('{"value":17}')]

    def provider(request):
        provider_calls.append(request)
        return responses.pop(0)

    runner = model_operation.DynamicModelOperationRunner(
        task_root=tmp_path,
        operation_instance_id="dynamic-valid-after-invalid",
        operation="hypothesis_generation",
        base_input={"scientific_question": "验证有效回答历史。"},
        provider=provider,
        message_builder=_builder([]),
        transport_attempts=1,
    )
    stage = _json_stage()
    slot = runner.reserve_model_stages([stage])[0]
    original_write_staging = runner._write_staging
    write_calls = [0]

    def fail_on_second_write(manifest):
        write_calls[0] += 1
        if write_calls[0] == 2:
            raise OSError("注入有效回答 staging 写盘失败")
        return original_write_staging(manifest)

    monkeypatch.setattr(runner, "_write_staging", fail_on_second_write)
    with pytest.raises(OSError, match="有效回答 staging 写盘失败"):
        runner.run_reserved_model_stage(slot, stage, {"query_angle": "有效"})
    assert len(provider_calls) == 2

    monkeypatch.setattr(runner, "_write_staging", original_write_staging)
    with pytest.raises(model_operation.ModelOperationError, match="不能执行两次"):
        runner.run_reserved_model_stage(slot, stage, {"query_angle": "有效"})
    finalized = runner.finalize_model_operation()

    assert finalized["status"] == "validated"
    assert len(provider_calls) == 2
    attempts = finalized["manifest"]["stages"][0]["attempts"]
    assert [attempt["attempt_index"] for attempt in attempts] == [0, 1]
    assert [attempt["validation_status"] for attempt in attempts] == [
        "invalid",
        "valid",
    ]


def test_dynamic_transport_after_invalid_retries(
    tmp_path,
    monkeypatch,
):
    """传输失败写盘异常只撤销传输 attempt，重试仍沿用 invalid 前缀。"""

    provider_calls = []
    builder_calls = []
    responses = [
        _envelope("not-json"),
        ConnectionError("受控传输失败"),
        _envelope('{"value":18}'),
    ]

    def provider(request):
        provider_calls.append(request)
        response = responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    runner = model_operation.DynamicModelOperationRunner(
        task_root=tmp_path,
        operation_instance_id="dynamic-transport-after-invalid",
        operation="hypothesis_generation",
        base_input={"scientific_question": "验证传输恢复。"},
        provider=provider,
        message_builder=_builder(builder_calls),
        transport_attempts=1,
    )
    stage = _json_stage()
    slot = runner.reserve_model_stages([stage])[0]
    original_write_staging = runner._write_staging
    write_calls = [0]

    def fail_on_second_write(manifest):
        write_calls[0] += 1
        if write_calls[0] == 2:
            raise OSError("注入 transport staging 写盘失败")
        return original_write_staging(manifest)

    monkeypatch.setattr(runner, "_write_staging", fail_on_second_write)
    with pytest.raises(OSError, match="transport staging 写盘失败"):
        runner.run_reserved_model_stage(slot, stage, {"query_angle": "传输"})
    assert len(provider_calls) == 2

    monkeypatch.setattr(runner, "_write_staging", original_write_staging)
    assert runner.run_reserved_model_stage(
        slot,
        stage,
        {"query_angle": "传输"},
    ) == {"value": 18}
    finalized = runner.finalize_model_operation()

    assert finalized["status"] == "validated"
    assert len(provider_calls) == 3
    attempts = finalized["manifest"]["stages"][0]["attempts"]
    assert [attempt["attempt_index"] for attempt in attempts] == [0, 1]
    assert [attempt["validation_status"] for attempt in attempts] == [
        "invalid",
        "valid",
    ]
    assert builder_calls[2]["protocol_error"]["failure_kind"] == "json_syntax"


def test_dynamic_failed_slot_cannot_start_again(tmp_path):
    """协议失败写入失败清单后，同一 slot 不能再次启动。"""

    provider_calls = []

    def provider(request):
        provider_calls.append(request)
        return _envelope("not-json")

    runner = model_operation.DynamicModelOperationRunner(
        task_root=tmp_path,
        operation_instance_id="dynamic-failed-slot",
        operation="hypothesis_generation",
        base_input={"scientific_question": "验证失败 slot。"},
        provider=provider,
        message_builder=_builder([]),
        transport_attempts=1,
    )
    stage = _json_stage()
    slot = runner.reserve_model_stages([stage])[0]

    with pytest.raises(model_operation.ModelOperationStageError, match="json_syntax"):
        runner.run_reserved_model_stage(slot, stage, {"query_angle": "失败"})
    with pytest.raises(model_operation.ModelOperationError, match="不能执行两次"):
        runner.run_reserved_model_stage(slot, stage, {"query_angle": "失败"})
    assert len(provider_calls) == model_operation.PROTOCOL_RESPONSE_LIMIT
    finalized = runner.finalize_model_operation()
    assert finalized["status"] == "failed"
    assert len(finalized["manifest"]["stages"][0]["attempts"]) == (
        model_operation.PROTOCOL_RESPONSE_LIMIT
    )


def test_dynamic_regeneration_remembers_all_prior_safe_errors(tmp_path):
    """动态假设子阶段重答时不能在两类已知格式错误之间反复摆动。"""

    responses = [
        _envelope('{"value":1,"value":2}'),
        _envelope('{"value":"wrong"}'),
        _envelope('{"value":2}'),
    ]
    builder_calls = []
    runner = model_operation.DynamicModelOperationRunner(
        task_root=tmp_path,
        operation_instance_id="dynamic-cumulative-errors",
        operation="hypothesis_generation",
        base_input={"scientific_question": "验证累计错误反馈。"},
        provider=lambda _request: responses.pop(0),
        message_builder=_builder(builder_calls),
    )

    value = runner.run_model_stage(
        _json_stage(),
        {"query_angle": "受控研究角度"},
    )
    result = runner.finalize_model_operation()

    assert value == {"value": 2}
    assert result["status"] == "validated"
    assert builder_calls[2]["protocol_error"] == {
        "failure_kind": "schema",
        "summary": (
            'schema_violation.v1:[{"expected":"integer","observed":"string",'
            '"path":"/value","rule":"type"}]; earlier stage failures must not '
            'recur: json_syntax: json_syntax.v1:{"column":0,"line":0,'
            '"reason":"duplicate object field"}'
        ),
    }


def test_schema_regeneration_reports_safe_length_coordinates(tmp_path):
    """长度失败应告诉模型具体位置，但不能把原始正文或本机路径带进重答提示。"""

    long_statement = "甲" * 305 + _ABSOLUTE_PRIVATE_EVIDENCE
    stage = model_operation.ModelStage(
        stage="hypothesis",
        stage_kind="model",
        validation_kind="json_schema",
        query_index=1,
        schema={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "statement": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 300,
                }
            },
            "required": ["statement"],
        },
    )
    calls = []
    runner, _, builder_calls = _runner(
        tmp_path,
        [
            _envelope(json.dumps({"statement": long_statement}, ensure_ascii=False)),
            _envelope('{"statement":"合格"}'),
        ],
        [stage],
        call_log=calls,
    )

    result = runner.run_model_operation()

    expected_summary = (
        "schema_violation.v1:"
        '[{"expected":300,"observed":326,'
        '"path":"/statement","rule":"maxLength"}]'
    )
    assert result["status"] == "validated"
    assert builder_calls[1]["protocol_error"] == {
        "failure_kind": "schema",
        "summary": expected_summary,
    }
    serialized_calls = json.dumps(builder_calls, ensure_ascii=False)
    assert long_statement not in serialized_calls
    assert _ABSOLUTE_PRIVATE_EVIDENCE not in serialized_calls
    first_attempt = result["manifest"]["stages"][0]["attempts"][0]
    error = _read_ref(tmp_path, first_attempt["error_ref"])
    assert error["summary"] == expected_summary


def test_transport_retry_is_counted_separately_from_protocol_responses(tmp_path):
    runner, provider_calls, _ = _runner(
        tmp_path,
        [ConnectionError("secret endpoint 1"), ConnectionError("secret endpoint 2"), _envelope('{"value":3}')],
        [_json_stage()],
        transport_attempts=3,
    )

    result = runner.run_model_operation()

    assert result["status"] == "validated"
    assert result["values"] == [{"value": 3}]
    assert len(provider_calls) == 3
    assert len(result["manifest"]["stages"][0]["attempts"]) == 1


def test_exhausted_transport_failure_stops_without_protocol_regeneration(tmp_path):
    runner, provider_calls, builder_calls = _runner(
        tmp_path,
        [ConnectionError("one"), TimeoutError("two"), RuntimeError("three")],
        [_json_stage()],
        transport_attempts=3,
    )

    result = runner.run_model_operation()

    assert result["status"] == "failed"
    assert result["failure_kind"] == "transport"
    assert len(provider_calls) == 3
    assert len(builder_calls) == 1
    attempt = result["manifest"]["stages"][0]["attempts"][0]
    assert attempt["response_ref"] is None
    assert attempt["provider_trace_id"] is None
    assert attempt["validation_status"] == "transport_failed"
    error = _read_ref(tmp_path, attempt["error_ref"])
    assert error == {
        "failure_kind": "transport",
        "finish_reason": None,
        "summary": "provider transport attempts were exhausted",
    }


def test_operation_writes_strict_staging_and_content_addressed_manifest(tmp_path):
    runner, _, _ = _runner(
        tmp_path,
        [_envelope('{"value":4}')],
        [_json_stage()],
    )

    result = runner.run_model_operation()

    operation_dir = (
        tmp_path
        / "working_memory/hypothesis_generation_inputs/operation-1"
    )
    assert (operation_dir / "base_input.json").exists()
    assert (operation_dir / "manifest.staging.json").exists()
    manifest_path = Path(result["manifest_path"])
    manifest_bytes = manifest_path.read_bytes()
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    assert manifest_path.name == f"manifest_{digest}.json"
    assert manifest_path.parent == operation_dir
    assert set(result["manifest"]) == {
        "schema_version",
        "operation_instance_id",
        "operation",
        "base_input_ref",
        "prior_manifest_ref",
        "stages",
    }
    assert result["manifest"]["prior_manifest_ref"] is None
    base_input = _read_ref(tmp_path, result["manifest"]["base_input_ref"])
    assert base_input == {
        "scientific_question": "苯甲醛红外与拉曼光谱",
        "task_config": {"num_queries": 2},
    }
    assert not Path(result["manifest_ref"]["relative_path"]).is_absolute()
    assert ".." not in Path(result["manifest_ref"]["relative_path"]).parts


def test_retry_failed_operation_reuses_validated_prefix_and_original_base_input(
    tmp_path,
):
    hypothesis_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "array",
        "minItems": 1,
        "maxItems": 1,
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "strategy_name": {"type": "string", "minLength": 1},
                "reasoning": {"type": "string", "minLength": 1},
            },
            "required": ["strategy_name", "reasoning"],
        },
    }
    stages = [
        model_operation.ModelStage(
            stage="query",
            stage_kind="model",
            validation_kind="query_delimited_text",
            expected_count=2,
        ),
        model_operation.ModelStage(
            stage="hypothesis",
            stage_kind="model",
            validation_kind="json_schema",
            query_index=1,
            schema=hypothesis_schema,
        ),
    ]
    first, first_provider_calls, _ = _runner(
        tmp_path,
        [
            _envelope("角度一<>角度二"),
            _envelope("invalid"),
            _envelope("invalid"),
            _envelope("invalid"),
            _envelope("invalid"),
        ],
        stages,
        operation_instance_id="operation-first",
    )
    failed = first.run_model_operation()
    assert failed["status"] == "failed"
    assert len(first_provider_calls) == 5

    retry_calls = []
    retry_responses = [
        _envelope("invalid", trace_id="trace-retry-invalid"),
        _envelope(
            '[{"reasoning":"可证伪预测","strategy_name":"新策略"}]',
            trace_id="trace-retry",
        ),
    ]

    def retry_provider(messages):
        retry_calls.append(messages)
        return retry_responses.pop(0)

    retry_builder_calls = []
    retry = model_operation.ModelOperationRunner.from_failed_manifest(
        task_root=tmp_path,
        operation_instance_id="operation-retry",
        prior_manifest_path=failed["manifest_path"],
        stages=stages,
        provider=retry_provider,
        message_builder=_builder(retry_builder_calls),
        transport_attempts=3,
    )
    recovered = retry.run_model_operation()

    assert recovered["status"] == "validated"
    assert recovered["values"] == [
        ["角度一", "角度二"],
        [{"reasoning": "可证伪预测", "strategy_name": "新策略"}],
    ]
    assert len(retry_calls) == 2
    assert retry_builder_calls[0]["retry_failed_operation"] is True
    assert retry_builder_calls[1]["retry_failed_operation"] is True
    assert retry_builder_calls[1]["protocol_error"] == {
        "failure_kind": "json_syntax",
        "summary": (
            'json_syntax.v1:{"column":1,"line":1,"reason":"syntax error"}'
        ),
    }
    assert recovered["manifest"]["prior_manifest_ref"] == failed["manifest_ref"]
    assert recovered["manifest"]["stages"][0] == failed["manifest"]["stages"][0]
    first_base = (
        tmp_path / failed["manifest"]["base_input_ref"]["relative_path"]
    ).read_bytes()
    retry_base = (
        tmp_path / recovered["manifest"]["base_input_ref"]["relative_path"]
    ).read_bytes()
    assert retry_base == first_base


def test_new_operation_does_not_scan_or_reuse_orphan_manifests(tmp_path):
    old, _, _ = _runner(
        tmp_path,
        [_envelope('{"value":1}')],
        [_json_stage()],
        operation_instance_id="operation-old",
    )
    old.run_model_operation()
    new, provider_calls, _ = _runner(
        tmp_path,
        [_envelope('{"value":2}')],
        [_json_stage()],
        operation_instance_id="operation-new",
    )

    result = new.run_model_operation()

    assert result["status"] == "validated"
    assert result["values"] == [{"value": 2}]
    assert len(provider_calls) == 1
    assert result["manifest"]["prior_manifest_ref"] is None


def test_gaussian_expert_text_is_not_forced_through_json_protocol():
    envelope = _envelope(
        "SCF 未收敛；建议先检查初猜和积分网格。",
        finish_reason="length",
    )

    assert model_operation.extract_gaussian_expert_text(envelope) == (
        "SCF 未收敛；建议先检查初猜和积分网格。"
    )
    assert model_operation.extract_gaussian_expert_text(_envelope("   ")) is None
    assert model_operation.extract_gaussian_expert_text({"choices": []}) is None


def test_runner_uses_the_shared_prompt_catalog_for_complete_regeneration(tmp_path):
    provider_messages = []
    responses = [
        _envelope("只有一个角度"),
        _envelope("光谱归属角度<>构象敏感性角度"),
    ]

    def provider(request):
        provider_messages.append(request["messages"])
        return responses.pop(0)

    runner = model_operation.ModelOperationRunner(
        task_root=tmp_path,
        operation_instance_id="catalog-operation",
        operation="hypothesis_generation",
        base_input={
            "scientific_question": "苯甲醛红外与拉曼光谱",
            "effective_chemistry_context": {
                "target_species": "benzaldehyde"
            },
            "num_queries": 2,
            "retrieval_projection": {
                "keywords": ["benzaldehyde vibrational spectra"],
                "literature_review": "证据有限。",
                "mechanistic_clues": [],
                "limitations": ["没有实验峰表"],
                "chemistry_context": {"charge": 0, "multiplicity": 1},
            },
        },
        stages=[
            model_operation.ModelStage(
                stage="query",
                stage_kind="model",
                validation_kind="query_delimited_text",
                expected_count=2,
            )
        ],
        provider=provider,
        transport_attempts=1,
    )

    result = runner.run_model_operation()

    assert result["status"] == "validated"
    assert result["values"] == [["光谱归属角度", "构象敏感性角度"]]
    assert "字面量 <>" in provider_messages[0][0]["content"]
    assert "query_delimited_text" not in provider_messages[0][0]["content"]
    first_envelope = json.loads(provider_messages[0][1]["content"])
    second_envelope = json.loads(provider_messages[1][1]["content"])
    assert first_envelope["protocol_error"] is None
    assert second_envelope["protocol_error"]["failure_kind"] == "domain"
    assert second_envelope["protocol_error"]["summary"].startswith(
        "domain_violation.v1:"
    )
    assert "需要精确返回 2 个" in second_envelope["protocol_error"]["summary"]
    assert "实际得到 1 个" in second_envelope["protocol_error"]["summary"]
    assert "只有一个角度" not in provider_messages[1][1]["content"]
    assert second_envelope["operation_retry"] is False


def test_dynamic_operation_records_retrieval_and_reserved_stage_order(tmp_path):
    """并行子调用即使逆序完成，清单也必须保持调用方预留的逻辑顺序。"""

    def provider(request):
        messages = request["messages"]
        payload = json.loads(messages[-1]["content"])
        return _envelope(payload["winner"], trace_id=f"trace-{payload['winner']}")

    runner = model_operation.DynamicModelOperationRunner(
        task_root=tmp_path,
        operation_instance_id="dynamic-order",
        operation="hypothesis_generation",
        base_input={"scientific_question": "验证动态子调用顺序。"},
        provider=provider,
        message_builder=_builder([]),
    )
    retrieval = runner.run_retrieval_stage(
        request={"question": "验证动态子调用顺序。"},
        retrieve=lambda _receipt_sink: {"literature_review": "受控背景"},
    )
    rank_stage = model_operation.ModelStage(
        stage="rank",
        stage_kind="model",
        validation_kind="text_enum_rank",
    )
    slots = runner.reserve_model_stages([rank_stage, rank_stage])
    second = runner.run_reserved_model_stage(
        slots[1],
        rank_stage,
        {"winner": "B"},
    )
    first = runner.run_reserved_model_stage(
        slots[0],
        rank_stage,
        {"winner": "A"},
    )
    result = runner.finalize_model_operation()

    assert retrieval == {"literature_review": "受控背景"}
    assert first == "A"
    assert second == "B"
    assert result["status"] == "validated"
    assert [item["stage"] for item in result["manifest"]["stages"]] == [
        "retrieval",
        "rank",
        "rank",
    ]
    assert [item["stage_index"] for item in result["manifest"]["stages"]] == [
        1,
        2,
        3,
    ]


def test_dynamic_retrieval_response_preserves_each_inner_model_usage(tmp_path):
    """Retrieval 内部模型调用必须按真实 attempt 落盘，不能只保存聚合结果。"""

    runner = model_operation.DynamicModelOperationRunner(
        task_root=tmp_path,
        operation_instance_id="dynamic-retrieval-usage",
        operation="hypothesis_generation",
        base_input={"scientific_question": "记录检索模型用量。"},
        provider=lambda _request: _envelope("unused"),
        message_builder=_builder([]),
    )

    def retrieve(record_model_call):
        record_model_call(
            {
                "stage": "keyword",
                "transport_attempt_index": 0,
                "request": {"compiled": "request"},
                "response": _envelope(
                    '["benzaldehyde", "DFT", "IR", "Raman", "frequency"]',
                    trace_id="retrieval-trace",
                ),
                "validation_status": "valid",
                "failure_kind": None,
                "summary": None,
            }
        )
        return {"literature_review": "受控背景"}

    assert runner.run_retrieval_stage(
        request={"question": "记录检索模型用量。"},
        retrieve=retrieve,
    ) == {"literature_review": "受控背景"}
    finalized = runner.finalize_model_operation()
    attempt = finalized["manifest"]["stages"][0]["attempts"][0]
    wrapper = json.loads(
        (tmp_path / attempt["response_ref"]["relative_path"]).read_text(
            encoding="utf-8"
        )
    )

    assert wrapper["schema_version"] == 1
    assert wrapper["result"] == {"literature_review": "受控背景"}
    assert len(wrapper["model_calls"]) == 1
    receipt = wrapper["model_calls"][0]
    assert receipt["stage"] == "keyword"
    assert receipt["provider_trace_id"] == "retrieval-trace"
    saved_response = json.loads(
        (tmp_path / receipt["response_ref"]["relative_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert saved_response["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
    }


def test_failed_retrieval_keeps_invalid_model_response_usage(tmp_path):
    """检索正文不合格时也要保留响应和 usage，不能只留下外层错误。"""

    runner = model_operation.DynamicModelOperationRunner(
        task_root=tmp_path,
        operation_instance_id="dynamic-retrieval-invalid-usage",
        operation="hypothesis_generation",
        base_input={"scientific_question": "记录失败检索用量。"},
        provider=lambda _request: _envelope("unused"),
        message_builder=_builder([]),
    )

    def retrieve(record_model_call):
        record_model_call(
            {
                "stage": "keyword",
                "transport_attempt_index": 0,
                "request": {"compiled": "request"},
                "response": _envelope(
                    "not-json",
                    trace_id="retrieval-invalid-trace",
                ),
                "validation_status": "invalid",
                "failure_kind": "json_syntax",
                "summary": "关键词响应不是合法 JSON",
            }
        )
        error = RuntimeError("关键词协议失败")
        error.failure_kind = "json_syntax"
        raise error

    with pytest.raises(
        model_operation.ModelOperationStageError,
        match="retrieval",
    ):
        runner.run_retrieval_stage(
            request={"question": "记录失败检索用量。"},
            retrieve=retrieve,
        )

    finalized = runner.finalize_model_operation()
    assert finalized["status"] == "failed"
    assert finalized["failure_kind"] == "json_syntax"
    attempt = finalized["manifest"]["stages"][0]["attempts"][0]
    assert attempt["validation_status"] == "invalid"
    wrapper = json.loads(
        (tmp_path / attempt["response_ref"]["relative_path"]).read_text(
            encoding="utf-8"
        )
    )
    receipt = wrapper["model_calls"][0]
    saved_response = json.loads(
        (tmp_path / receipt["response_ref"]["relative_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert saved_response["usage"]["total_tokens"] == 30


def test_regeneration_memory_switch_is_not_forwarded_to_relationship_stage(tmp_path):
    """再生成结束后，关系判断不应继续收到 Memory 开关。"""

    provider_messages = []

    def provider(request):
        messages = request["messages"]
        provider_messages.append(messages)
        if "假设再生成员" in messages[0]["content"]:
            return _envelope('{"value":1}')
        return _envelope("independent")

    runner = model_operation.DynamicModelOperationRunner(
        task_root=tmp_path,
        operation_instance_id="dynamic-regeneration-memory-boundary",
        operation="hypothesis_generation",
        base_input={"scientific_question": "验证再生成后的提示边界。"},
        provider=provider,
        memory_enabled=False,
    )
    regeneration_stage = model_operation.ModelStage(
        stage="regeneration",
        stage_kind="model",
        validation_kind="json_schema",
        schema={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    )
    relationship_stage = model_operation.ModelStage(
        stage="relationship",
        stage_kind="model",
        validation_kind="text_enum_relationship",
    )

    assert runner.run_model_stage(regeneration_stage, {}) == {"value": 1}
    assert runner.run_model_stage(
        relationship_stage,
        {
            "scientific_context": {"scientific_question": "比较两个假设。"},
            "hypothesis_a": {"strategy_name": "甲", "reasoning": "论证甲"},
            "hypothesis_b": {"strategy_name": "乙", "reasoning": "论证乙"},
        },
    ) == "independent"
    result = runner.finalize_model_operation()

    assert result["status"] == "validated"
    assert len(provider_messages) == 2


def test_dynamic_operation_retry_reuses_validated_prefix(tmp_path):
    """假设生成重试应复用 Retrieval 和 query，只重做首个失败子调用。"""

    query_stage = model_operation.ModelStage(
        stage="query",
        stage_kind="model",
        validation_kind="query_delimited_text",
        expected_count=1,
    )
    hypothesis_stage = _json_stage()
    first_responses = [
        _envelope("受控角度"),
        _envelope("invalid"),
        _envelope("invalid"),
        _envelope("invalid"),
        _envelope("invalid"),
    ]

    first = model_operation.DynamicModelOperationRunner(
        task_root=tmp_path,
        operation_instance_id="dynamic-first",
        operation="hypothesis_generation",
        base_input={"scientific_question": "验证前缀恢复。"},
        provider=lambda _messages: first_responses.pop(0),
        message_builder=_builder([]),
    )
    first.run_retrieval_stage(
        request={"question": "验证前缀恢复。"},
        retrieve=lambda _receipt_sink: {"literature_review": "冻结背景"},
    )
    assert first.run_model_stage(query_stage, {"query": "原始问题"}) == [
        "受控角度"
    ]
    with pytest.raises(model_operation.ModelOperationStageError):
        first.run_model_stage(hypothesis_stage, {"query_angle": "受控角度"})
    failed = first.finalize_model_operation()
    assert failed["status"] == "failed"

    retry_calls = []
    retry = model_operation.DynamicModelOperationRunner.from_failed_manifest(
        task_root=tmp_path,
        operation_instance_id="dynamic-retry",
        prior_manifest_path=failed["manifest_path"],
        provider=lambda messages: (
            retry_calls.append(messages) or _envelope('{"value":2}')
        ),
        message_builder=_builder([]),
    )
    assert retry.run_retrieval_stage(
        request={"question": "验证前缀恢复。"},
        retrieve=lambda _receipt_sink: (_ for _ in ()).throw(
            AssertionError("已验证 Retrieval 不应再次执行")
        ),
    ) == {"literature_review": "冻结背景"}
    assert retry.run_model_stage(query_stage, {"query": "原始问题"}) == [
        "受控角度"
    ]
    assert retry.run_model_stage(
        hypothesis_stage,
        {"query_angle": "受控角度"},
    ) == {"value": 2}
    recovered = retry.finalize_model_operation()

    assert recovered["status"] == "validated"
    assert len(retry_calls) == 1
    assert recovered["manifest"]["prior_manifest_ref"] == failed["manifest_ref"]
    assert recovered["manifest"]["stages"][:2] == failed["manifest"]["stages"][:2]


def test_dynamic_retry_replays_protocol_errors_before_valid_prefix(tmp_path):
    """前缀若在第二次回答才成功，恢复时必须重建首答错误后的精确请求。"""

    first_stage = _json_stage()
    second_stage = model_operation.ModelStage(
        stage="hypothesis",
        stage_kind="model",
        validation_kind="json_schema",
        query_index=2,
        schema=first_stage.schema,
    )

    def sensitive_builder(
        operation,
        base_input,
        stage=None,
        protocol_error=None,
        memory_enabled=None,
        retry_failed_operation=False,
        expert_fallback=False,
    ):
        return [
            {
                "role": "system",
                "content": json.dumps(
                    {
                        "operation": operation,
                        "stage": stage,
                        "protocol_error": protocol_error,
                        "retry_failed_operation": retry_failed_operation,
                    },
                    sort_keys=True,
                ),
            },
            {"role": "user", "content": json.dumps(base_input, sort_keys=True)},
        ]

    first_responses = [
        _envelope('{"value":"wrong"}'),
        _envelope('{"value":1}'),
        *[_envelope("invalid") for _ in range(4)],
    ]
    first = model_operation.DynamicModelOperationRunner(
        task_root=tmp_path,
        operation_instance_id="dynamic-prefix-retry-first",
        operation="hypothesis_generation",
        base_input={"scientific_question": "验证带错误历史的前缀恢复。"},
        provider=lambda _request: first_responses.pop(0),
        message_builder=sensitive_builder,
    )
    assert first.run_model_stage(first_stage, {"query_angle": "角度一"}) == {
        "value": 1
    }
    with pytest.raises(model_operation.ModelOperationStageError):
        first.run_model_stage(second_stage, {"query_angle": "角度二"})
    failed = first.finalize_model_operation()

    retry_calls = []
    retry = model_operation.DynamicModelOperationRunner.from_failed_manifest(
        task_root=tmp_path,
        operation_instance_id="dynamic-prefix-retry-second",
        prior_manifest_path=failed["manifest_path"],
        provider=lambda request: (
            retry_calls.append(request) or _envelope('{"value":2}')
        ),
        message_builder=sensitive_builder,
    )
    assert retry.run_model_stage(first_stage, {"query_angle": "角度一"}) == {
        "value": 1
    }
    assert retry.run_model_stage(second_stage, {"query_angle": "角度二"}) == {
        "value": 2
    }
    recovered = retry.finalize_model_operation()

    assert recovered["status"] == "validated"
    assert len(retry_calls) == 1


def test_fixed_runner_replays_finalized_success_without_provider(tmp_path):
    """成功 finalized manifest 可独立重放，且重放不产生新的 provider 调用。"""

    calls = []
    stage = _json_stage()
    base_input = {"scientific_question": "验证成功清单重放。"}

    def provider(request):
        calls.append(request)
        return _envelope('{"value": 11}')

    first = model_operation.ModelOperationRunner(
        task_root=tmp_path,
        operation_instance_id="fixed-replay-success",
        operation="hypothesis_generation",
        base_input=base_input,
        stages=[stage],
        provider=provider,
    )
    result = first.run_model_operation()
    replayed = model_operation.ModelOperationRunner.replay_finalized_manifest(
        task_root=tmp_path,
        manifest_path=result["manifest_path"],
        operation="hypothesis_generation",
        operation_instance_id="fixed-replay-success",
        expected_base_input=base_input,
        stages=[stage],
    )

    assert len(calls) == 1
    assert replayed["status"] == "validated"
    assert replayed["failure_kind"] is None
    assert replayed["values"] == [{"value": 11}]
    assert replayed["manifest_ref"] == result["manifest_ref"]


def test_fixed_runner_replays_finalized_failure_without_provider(tmp_path):
    """失败 finalized manifest 重放时保留失败类别，不重新发送四次请求。"""

    calls = []
    stage = _json_stage()
    base_input = {"scientific_question": "验证失败清单重放。"}

    def provider(request):
        calls.append(request)
        return _envelope('{"wrong": true}')

    first = model_operation.ModelOperationRunner(
        task_root=tmp_path,
        operation_instance_id="fixed-replay-failure",
        operation="hypothesis_generation",
        base_input=base_input,
        stages=[stage],
        provider=provider,
    )
    result = first.run_model_operation()
    assert result["status"] == "failed"
    assert result["failure_kind"] == "schema"

    replayed = model_operation.ModelOperationRunner.replay_finalized_manifest(
        task_root=tmp_path,
        manifest_path=result["manifest_path"],
        operation="hypothesis_generation",
        operation_instance_id="fixed-replay-failure",
        expected_base_input=base_input,
        stages=[stage],
    )

    assert len(calls) == 4
    assert replayed["status"] == "failed"
    assert replayed["failure_kind"] == "schema"
    assert replayed["values"] == []
    assert replayed["manifest_ref"] == result["manifest_ref"]


def test_fixed_runner_operation_lookup_does_not_scan_orphan_manifests(tmp_path):
    """确定 operation 目录没有清单时，其他目录的 finalized 文件不会被误用。"""

    stage = _json_stage()
    base_input = {"scientific_question": "验证孤立清单隔离。"}
    first = model_operation.ModelOperationRunner(
        task_root=tmp_path,
        operation_instance_id="orphan-operation",
        operation="hypothesis_generation",
        base_input=base_input,
        stages=[stage],
        provider=lambda _request: _envelope('{"value": 1}'),
    )
    orphan = first.run_model_operation()

    assert model_operation.ModelOperationRunner.replay_finalized_manifest_in_operation(
        task_root=tmp_path,
        operation="hypothesis_generation",
        operation_instance_id="different-operation",
        expected_base_input=base_input,
        stages=[stage],
    ) is None
    assert Path(orphan["manifest_path"]).is_file()


def test_fixed_runner_restarts_pristine_staging_before_request_is_written(tmp_path):
    """消息组装失败且尚未写请求时，同一 operation 可以安全重试。"""

    stage = _json_stage()
    base_input = {"scientific_question": "验证请求落盘前的安全恢复。"}

    def fail_before_request(*_args, **_kwargs):
        raise RuntimeError("模拟消息组装失败")

    interrupted = model_operation.ModelOperationRunner(
        task_root=tmp_path,
        operation_instance_id="pristine-staging-operation",
        operation="hypothesis_generation",
        base_input=base_input,
        stages=[stage],
        provider=lambda _request: pytest.fail("消息组装失败后不应调用 provider"),
        message_builder=fail_before_request,
    )
    with pytest.raises(RuntimeError, match="消息组装失败"):
        interrupted.run_model_operation()

    operation_dir = (
        tmp_path
        / "working_memory/hypothesis_generation_inputs/pristine-staging-operation"
    )
    assert sorted(path.name for path in operation_dir.iterdir()) == [
        "base_input.json",
        "manifest.staging.json",
    ]
    assert model_operation.ModelOperationRunner.replay_finalized_manifest_in_operation(
        task_root=tmp_path,
        operation="hypothesis_generation",
        operation_instance_id="pristine-staging-operation",
        expected_base_input=base_input,
        stages=[stage],
    ) is None

    calls = []
    resumed = model_operation.ModelOperationRunner(
        task_root=tmp_path,
        operation_instance_id="pristine-staging-operation",
        operation="hypothesis_generation",
        base_input=base_input,
        stages=[stage],
        provider=lambda request: calls.append(request) or _envelope('{"value":12}'),
        message_builder=_builder([]),
    ).run_model_operation()

    assert resumed["status"] == "validated"
    assert resumed["values"] == [{"value": 12}]
    assert len(calls) == 1


def test_fixed_runner_does_not_restart_after_request_is_written(tmp_path):
    """请求已写盘但 provider 结果未知时，恢复必须继续失败关闭。"""

    stage = _json_stage()
    base_input = {"scientific_question": "验证请求发出后的不确定状态。"}

    def interrupt_after_request(_request):
        raise KeyboardInterrupt("模拟 provider 调用期间中断")

    interrupted = model_operation.ModelOperationRunner(
        task_root=tmp_path,
        operation_instance_id="request-written-operation",
        operation="hypothesis_generation",
        base_input=base_input,
        stages=[stage],
        provider=interrupt_after_request,
        message_builder=_builder([]),
    )
    with pytest.raises(KeyboardInterrupt, match="provider 调用期间中断"):
        interrupted.run_model_operation()

    operation_dir = (
        tmp_path
        / "working_memory/hypothesis_generation_inputs/request-written-operation"
    )
    assert list((operation_dir / "stages").rglob("attempt_0_request_*.json"))
    with pytest.raises(
        model_operation.ModelOperationError,
        match="对应 finalized manifest 缺失",
    ):
        model_operation.ModelOperationRunner.replay_finalized_manifest_in_operation(
            task_root=tmp_path,
            operation="hypothesis_generation",
            operation_instance_id="request-written-operation",
            expected_base_input=base_input,
            stages=[stage],
        )
