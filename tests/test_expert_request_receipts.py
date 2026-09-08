"""专家请求收据、消息摘要和恢复去重边界测试。"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from chemistry_multiagent.utils.expert_request_receipts import (
    ExpertRequestReceiptError,
    build_expert_request_dispatch_observer,
    build_expert_request_receipt_bundle,
    find_started_expert_request_receipt,
    hash_expert_messages,
    hash_expert_request_descriptor,
    write_expert_request_terminal_receipt,
)
from chemistry_multiagent.memory.contracts import (
    validate_expert_request_receipt,
)
from chemistry_multiagent.utils.task_artifacts import (
    TaskArtifactError,
    write_task_relative_json,
)


def _combination() -> dict[str, Any]:
    """返回请求收据所需的最小组合身份。"""

    return {
        "combination_id": "combination-1",
        "workflow_version": 1,
        "execution_id": "execution-1",
    }


def test_started_receipt_blocks_crash_retry_until_terminal_marker(tmp_path) -> None:
    """provider 已可能接收请求而进程崩溃时，恢复不得重新提交同一请求。"""

    context = SimpleNamespace(task_root=tmp_path)
    combination = _combination()
    descriptor = {"phase": "gaussian_step_review", "input": {"step_id": "s1"}}
    compiled_ref, request_receipt = build_expert_request_receipt_bundle(
        context,
        combination=combination,
        operation_instance_id="expert-review-1",
        phase="gaussian_step_review",
        request_descriptor=descriptor,
        backend_fingerprint="a" * 64,
    )
    assert request_receipt["status"] == "compiled"
    assert request_receipt["calls"] == []
    assert not list((tmp_path / "working_memory" / "expert_requests").rglob("started_request_*.json"))
    observer = build_expert_request_dispatch_observer(
        context,
        request_receipt=request_receipt,
    )
    observer(
        "gaussian_step_review",
        "expert",
        [
            {"role": "system", "content": "review"},
            {"role": "user", "content": "{\"step\":1}"},
        ],
    )
    started_ref = request_receipt["started_ref"]
    assert isinstance(started_ref, str)
    request_hash = hash_expert_request_descriptor(descriptor)

    pending = find_started_expert_request_receipt(
        context,
        combination=combination,
        phase="gaussian_step_review",
        request_hash=request_hash,
    )
    assert pending is not None
    assert pending[0] == started_ref

    terminal_ref = write_expert_request_terminal_receipt(
        context,
        started_receipt_ref=started_ref,
        request_receipt=request_receipt,
        status="failed",
    )
    assert terminal_ref is not None
    assert request_receipt["status"] == "failed"
    assert (
        find_started_expert_request_receipt(
            context,
            combination=combination,
            phase="gaussian_step_review",
            request_hash=request_hash,
        )
        is None
    )


def test_debate_receipt_keeps_three_call_hashes_and_terminal_recovery(tmp_path) -> None:
    """Debate 的三次 provider 调用共享逻辑 operation，但各自保留稳定请求摘要。"""

    context = SimpleNamespace(task_root=tmp_path)
    combination = _combination()
    descriptor = {
        "phase": "debate",
        "workflow": {"steps": ["s1"]},
        "evidence": ["e1"],
    }
    compiled_ref, request_receipt = build_expert_request_receipt_bundle(
        context,
        combination=combination,
        operation_instance_id="debate-1",
        phase="debate",
        request_descriptor=descriptor,
        backend_fingerprint="b" * 64,
    )
    observer = build_expert_request_dispatch_observer(
        context,
        request_receipt=request_receipt,
    )
    for operation, role, content in (
        ("workflow_expert_opinion", "expert", "expert"),
        ("workflow_general_opinion", "general", "general"),
        ("debate_judge", "general", "judge"),
    ):
        observer(
            operation,
            role,
            [
                {"role": "system", "content": operation},
                {"role": "user", "content": content},
            ],
        )
    started_ref = request_receipt["started_ref"]
    assert len(request_receipt["calls"]) == 3
    assert len({item["request_hash"] for item in request_receipt["calls"]}) == 3
    assert validate_expert_request_receipt(request_receipt)["started_ref"] == started_ref
    request_hash = hash_expert_request_descriptor(descriptor)
    assert (
        find_started_expert_request_receipt(
            context,
            combination=combination,
            phase="debate",
            request_hash=request_hash,
        )
        is not None
    )
    write_expert_request_terminal_receipt(
        context,
        started_receipt_ref=started_ref,
        request_receipt=request_receipt,
        status="completed",
    )
    assert request_receipt["status"] == "completed"
    assert (
        find_started_expert_request_receipt(
            context,
            combination=combination,
            phase="debate",
            request_hash=request_hash,
        )
        is None
    )


def test_receipt_records_hash_before_provider_without_message_body(tmp_path) -> None:
    """dispatch observer 在 provider 前保存实际消息摘要，但不保存消息正文。"""

    context = SimpleNamespace(task_root=tmp_path)
    compiled_ref, receipt = build_expert_request_receipt_bundle(
        context,
        combination=_combination(),
        operation_instance_id="expert-review-final-message",
        phase="gaussian_step_review",
        request_descriptor={"step_ref": "s1"},
        backend_fingerprint="c" * 64,
    )
    messages = [
        {"role": "system", "content": "review"},
        {"role": "user", "content": '{"route":"#p opt"}'},
    ]
    observer = build_expert_request_dispatch_observer(
        context,
        request_receipt=receipt,
    )
    observer("gaussian_step_review", "expert", messages)
    started_ref = receipt["started_ref"]
    digest = hash_expert_messages(messages)
    assert receipt["calls"][0]["message_hash"] == digest
    terminal_ref = write_expert_request_terminal_receipt(
        context,
        started_receipt_ref=started_ref,
        request_receipt=receipt,
        status="completed",
    )
    assert terminal_ref is not None
    validated = validate_expert_request_receipt(receipt)
    assert validated["calls"][0]["message_hash"] == digest
    serialized = str(validated)
    assert "#p opt" not in serialized


def test_second_crash_scan_returns_all_actual_calls(tmp_path) -> None:
    """第二笔实际调用崩溃时，扫描结果必须包含此前和当前两笔 call。"""

    context = SimpleNamespace(task_root=tmp_path)
    descriptor = {"phase": "gaussian_step_review", "input": {"step_id": "s1"}}
    _, receipt = build_expert_request_receipt_bundle(
        context,
        combination=_combination(),
        operation_instance_id="expert-review-two-calls",
        phase="gaussian_step_review",
        request_descriptor=descriptor,
        backend_fingerprint="d" * 64,
    )
    observer = build_expert_request_dispatch_observer(context, request_receipt=receipt)
    observer("gaussian_step_review", "expert", [{"role": "user", "content": "one"}])
    observer("gaussian_step_review", "expert", [{"role": "user", "content": "two"}])
    pending = find_started_expert_request_receipt(
        context,
        combination=_combination(),
        phase="gaussian_step_review",
        request_hash=hash_expert_request_descriptor(descriptor),
    )
    assert pending is not None
    assert len(pending[1]["calls"]) == 2
    assert len({item["message_hash"] for item in pending[1]["calls"]}) == 2


def test_started_call_index_conflict_does_not_overwrite_prior_snapshot(tmp_path) -> None:
    """同一 call_index 的不同 started 快照必须拒绝覆盖先到达的事实。"""

    relative_path = "working_memory/expert_requests/op/started_request_0001.json"
    original = {"status": "started", "call_index": 0, "message_hash": "a"}
    replacement = {"status": "started", "call_index": 0, "message_hash": "b"}
    write_task_relative_json(tmp_path, relative_path, original)
    with pytest.raises(TaskArtifactError):
        write_task_relative_json(tmp_path, relative_path, replacement)
    assert json.loads((tmp_path / relative_path).read_text(encoding="utf-8")) == original


def test_receipt_observer_failure_prevents_provider_call(monkeypatch, tmp_path) -> None:
    """started 收据写入失败时，provider 调用次数必须保持零。"""

    context = SimpleNamespace(task_root=tmp_path)
    _, receipt = build_expert_request_receipt_bundle(
        context,
        combination=_combination(),
        operation_instance_id="expert-review-write-failure",
        phase="gaussian_step_review",
        request_descriptor={"step_ref": "s1"},
        backend_fingerprint="e" * 64,
    )
    observer = build_expert_request_dispatch_observer(context, request_receipt=receipt)
    calls: list[Any] = []

    def provider(_messages: Any) -> dict[str, str]:
        calls.append(True)
        return {"status": "approved", "reason": "ok"}

    from chemistry_multiagent.utils import expert_request_receipts as receipts_module

    def fail_write(*_args: Any, **_kwargs: Any) -> str:
        raise OSError("disk unavailable")

    monkeypatch.setattr(receipts_module, "write_task_relative_json", fail_write)
    from chemistry_multiagent.agents.expert_model_agent import ExpertModelAgent, ExpertModelConfig

    agent = ExpertModelAgent(
        ExpertModelConfig(enabled=True, expert_provider=provider),
    )
    with pytest.raises(ExpertRequestReceiptError):
        agent._call(
            "system",
            "gaussian_step_review",
            {"value": 1},
            role="expert",
            request_observer=observer,
        )
    assert calls == []


def test_terminal_write_failure_keeps_started_state(monkeypatch, tmp_path) -> None:
    """终局文件写失败时，内存和扫描结果仍保留 started。"""

    context = SimpleNamespace(task_root=tmp_path)
    descriptor = {"phase": "gaussian_step_review", "failure": True}
    _, receipt = build_expert_request_receipt_bundle(
        context,
        combination=_combination(),
        operation_instance_id="expert-review-terminal-write-failure",
        phase="gaussian_step_review",
        request_descriptor=descriptor,
        backend_fingerprint="1" * 64,
    )
    observer = build_expert_request_dispatch_observer(context, request_receipt=receipt)
    observer("gaussian_step_review", "expert", [{"role": "user", "content": "one"}])
    started_path = tmp_path / receipt["started_ref"]
    raw_started = json.loads(started_path.read_text(encoding="utf-8"))
    assert validate_expert_request_receipt(raw_started)["started_ref"] == receipt["started_ref"]

    from chemistry_multiagent.utils import expert_request_receipts as receipts_module

    def fail_terminal(*_args: Any, **_kwargs: Any) -> str:
        raise OSError("terminal disk unavailable")

    monkeypatch.setattr(receipts_module, "write_content_addressed_json", fail_terminal)
    with pytest.raises(OSError):
        write_expert_request_terminal_receipt(
            context,
            started_receipt_ref=receipt["started_ref"],
            request_receipt=receipt,
            status="failed",
        )
    assert receipt["status"] == "started"
    assert (
        find_started_expert_request_receipt(
            context,
            combination=_combination(),
            phase="gaussian_step_review",
            request_hash=hash_expert_request_descriptor(descriptor),
        )
        is not None
    )


def test_terminal_receipt_closes_all_started_snapshots(tmp_path) -> None:
    """completed 或 failed 终局收据都能关闭同一 operation 的全部 started 快照。"""

    for status in ("completed", "failed"):
        context = SimpleNamespace(task_root=tmp_path / status)
        context.task_root.mkdir()
        descriptor = {"phase": "gaussian_step_review", "status": status}
        _, receipt = build_expert_request_receipt_bundle(
            context,
            combination=_combination(),
            operation_instance_id=f"expert-review-terminal-{status}",
            phase="gaussian_step_review",
            request_descriptor=descriptor,
            backend_fingerprint="f" * 64,
        )
        observer = build_expert_request_dispatch_observer(context, request_receipt=receipt)
        observer("gaussian_step_review", "expert", [{"role": "user", "content": "one"}])
        observer("gaussian_step_review", "expert", [{"role": "user", "content": "two"}])
        write_expert_request_terminal_receipt(
            context,
            started_receipt_ref=receipt["started_ref"],
            request_receipt=receipt,
            status=status,
        )
        assert (
            find_started_expert_request_receipt(
                context,
                combination=_combination(),
                phase="gaussian_step_review",
                request_hash=hash_expert_request_descriptor(descriptor),
            )
            is None
        )


def test_role_backend_fingerprints_are_preserved_per_call(tmp_path) -> None:
    """7B expert 与通用 provider 指纹不同，收据按实际 role 分别记录。"""

    context = SimpleNamespace(task_root=tmp_path)
    _, receipt = build_expert_request_receipt_bundle(
        context,
        combination=_combination(),
        operation_instance_id="debate-role-fingerprints",
        phase="debate",
        request_descriptor={"phase": "debate"},
        backend_fingerprint="a" * 64,
    )
    observer = build_expert_request_dispatch_observer(
        context,
        request_receipt=receipt,
        backend_fingerprint_by_role={"expert": "a" * 64, "general": "b" * 64},
    )
    observer("workflow_expert_opinion", "expert", [{"role": "user", "content": "e"}])
    observer("workflow_general_opinion", "general", [{"role": "user", "content": "g"}])
    assert [item["backend_fingerprint"] for item in receipt["calls"]] == [
        "a" * 64,
        "b" * 64,
    ]


def test_fallback_role_backend_fingerprints_remain_distinct(tmp_path) -> None:
    """专家回退使用 Flash 指纹，通用意见仍使用 Pro 指纹。"""

    context = SimpleNamespace(task_root=tmp_path)
    _, receipt = build_expert_request_receipt_bundle(
        context,
        combination=_combination(),
        operation_instance_id="debate-fallback-fingerprint",
        phase="debate",
        request_descriptor={"phase": "debate"},
        backend_fingerprint="c" * 64,
    )
    observer = build_expert_request_dispatch_observer(
        context,
        request_receipt=receipt,
        backend_fingerprint_by_role={"expert": "c" * 64, "general": "b" * 64},
    )
    observer("workflow_expert_opinion", "expert", [{"role": "user", "content": "e"}])
    observer("workflow_general_opinion", "general", [{"role": "user", "content": "g"}])
    assert [item["backend_fingerprint"] for item in receipt["calls"]] == [
        "c" * 64,
        "b" * 64,
    ]
