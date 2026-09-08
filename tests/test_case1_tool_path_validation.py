"""Case 1 工具路径静态映射的确定性测试。"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import scripts.tasks.case1_validate_tool_path as case1_tool_path

from scripts.tasks.case1_validate_tool_path import (
    CASE1_NATIVE_CPU_CORES,
    CASE1_NATIVE_GPU_COUNT,
    CASE1_NATIVE_MEMORY_MB,
    CASE1_NATIVE_REPLAY_PLAN,
    STATIC_STEPS,
    calculate_action_request_fingerprint,
    attach_case1_analysis_outcome,
    build_case1_gaussian_deck,
    build_case1_fresh_analysis_program,
    build_case1_fresh_analysis_request,
    build_case1_native_request,
    build_case1_completeness_summary,
    case1_native_resource_limits,
    calculate_case1_delta_g_from_published_values,
    collect_case1_fresh_analysis_inputs,
    evaluate_case1_full_native_path,
    extract_native_checkpoint_path,
    extract_native_log_path,
    load_case1_summary,
    check_native_job_scientific_gate,
    build_native_operation_id,
    recover_existing_program_operation,
    restore_validation_incidents,
    run_case1_fresh_analysis,
    run_mcp_action_once,
    run_case1_native_step,
)


def _decorate_valid_native_record(
    tmp_path: Path,
    key: str,
    *,
    intent: str,
    parent: dict[str, str] | None = None,
    irc_completed: bool | None = None,
) -> dict:
    """为正向测试构造带收据、checkpoint 和日志哈希的 native 终态记录。"""

    slug = key.removeprefix("native_").replace("_", "-")
    operation_id = f"operation-{slug}"
    job_id = f"job-{slug}"
    checkpoint_path = f"outputs/{slug}.chk"
    log_path = f"outputs/{slug}.log"
    receipt_path = tmp_path / f"{operation_id}.json"
    receipt_path.write_text(
        json.dumps({"operation_id": operation_id, "job_id": job_id}),
        encoding="utf-8",
    )
    record = {
        "status": "completed",
        "ready": True,
        "job_id": job_id,
        "operation_id": operation_id,
        "operation_receipt_path": str(receipt_path),
        "calculation_intent": intent,
        "software_status": "normal_termination",
        "convergence_status": "converged",
        "scientific_validation_status": "mechanically_valid",
        "checkpoint_path": checkpoint_path,
        "log_path": log_path,
        "outputs": [
            {"path": checkpoint_path, "size_bytes": 128, "sha256": "a" * 64},
            {"path": log_path, "size_bytes": 256, "sha256": "b" * 64},
        ],
    }
    if parent is not None:
        record.update(
            {
                "parent_operation_id": parent["operation_id"],
                "parent_job_id": parent["job_id"],
                "parent_checkpoint_path": parent["checkpoint_path"],
            }
        )
    if irc_completed is not None:
        record["irc_bidirectional_completed"] = irc_completed
    return record


def _build_valid_native_runs(tmp_path: Path) -> dict[str, dict]:
    """构造两条路径的六个完整 native stage 测试记录。"""

    runs: dict[str, dict] = {}
    for label in ("TS-2a", "TS-2b-1"):
        step4_key = f"native_{label}_step4_opt_freq"
        step4 = _decorate_valid_native_record(
            tmp_path,
            step4_key,
            intent="transition_state",
        )
        runs[step4_key] = step4
        for stage, intent in (
            ("step7_irc", "reaction_path"),
            ("step10_single_point", "single_point"),
        ):
            key = f"native_{label}_{stage}"
            runs[key] = _decorate_valid_native_record(
                tmp_path,
                key,
                intent=intent,
                parent=step4,
                irc_completed=True if stage == "step7_irc" else None,
            )
    return runs


def test_official_composite_result_is_restored_only_from_matching_trace(
    tmp_path: Path,
) -> None:
    """同步 Action 只能用相同请求指纹的 trace/result 恢复，重复结果只记事故。"""

    action_id = "search_transition_state_path"
    backend_id = "transition_state_path_search"
    authored_request = {
        "action_id": action_id,
        "backend_id": backend_id,
        "inputs": {},
        "method_spec": {},
        "action_settings": {},
        "resource_limits": {"cpu_cores": 1, "memory_mb": 2048, "gpu_count": 1},
    }
    fingerprint = calculate_action_request_fingerprint(authored_request)
    result_root = tmp_path / "_tool_results"
    result_root.mkdir()

    wire_request = dict(authored_request)
    wire_request.pop("action_id")
    wire_request["component_backends"] = {}
    wire_request["source_id"] = None
    wrong_request = json.loads(json.dumps(wire_request))
    wrong_request["resource_limits"]["gpu_count"] = 2
    trace_records = [
        {
            "sequence": 1,
            "tool": action_id,
            "arguments": {"request": wrong_request},
            "result_path": "_tool_results/0001_search_transition_state_path.json",
        },
        {
            "sequence": 2,
            "tool": action_id,
            "arguments": {"request": wire_request},
            "result_path": "_tool_results/0002_search_transition_state_path.json",
        },
        {
            "sequence": 3,
            "tool": action_id,
            "arguments": {"request": wire_request},
            "result_path": "_tool_results/0003_search_transition_state_path.json",
        },
    ]
    (tmp_path / "_tool_trace.jsonl").write_text(
        "\n".join(json.dumps(item) for item in trace_records) + "\n",
        encoding="utf-8",
    )

    def write_result(sequence: int, artifact_id: str) -> None:
        value = {
            "status": "partial_success",
            "result": {
                "status": "partial_success",
                "action": action_id,
                "requested_backend": backend_id,
                "backend": backend_id,
                "result": {"artifact_id": artifact_id},
                "output_artifacts": [{"artifact_id": artifact_id}],
                "warnings": [f"result-{sequence}"],
            },
        }
        (result_root / f"{sequence:04d}_search_transition_state_path.json").write_text(
            json.dumps(value), encoding="utf-8"
        )

    write_result(1, "art_wrong")
    write_result(2, "art_first_matching")
    write_result(3, "art_duplicate_matching")
    manifest = {
        "executions": {
            "step1_official_composite_pipeline": {
                "status": "uncertain_requires_manual_recovery",
                "request_fingerprint": fingerprint,
                "result_locator": {
                    "action_id": action_id,
                    "backend_id": backend_id,
                },
            }
        },
        "validation_incidents": [
            {
                "incident_id": "official-composite-duplicate-0003_search_transition_state_path",
                "type": "accidental_duplicate_submission",
                "scope": "step1_official_composite_pipeline",
                "service_result": {"artifact_id": "art_duplicate_matching"},
            },
            {
                "incident_id": "interrupted-client",
                "type": "uncertain_sync_action",
                "scope": "step1_official_composite_pipeline",
            },
        ],
    }

    restore_validation_incidents(manifest, runtime_root=tmp_path)

    official = manifest["executions"]["step1_official_composite_pipeline"]
    assert official["status"] == "completed"
    assert official["minichem_status"] == "partial_success"
    assert official["request_fingerprint"] == fingerprint
    assert official["output_artifacts"] == [{"artifact_id": "art_first_matching"}]
    assert official["recovered_from_service_trace"] is True
    assert len(official["service_result_evidence"]) == 2
    assert official["service_result_evidence"][0]["sequence"] == 2
    incidents = manifest["validation_incidents"]
    assert len(incidents) == 2
    assert incidents[0]["attempt_number"] == 2
    assert incidents[0]["service_result"]["sequence"] == 3
    assert incidents[1]["resolution_status"] == "resolved_from_matching_trace"


def test_case1_static_mapping_has_eleven_ordered_steps() -> None:
    """论文附录的 11 步工作流顺序和关键工具名必须保持可审计。"""

    assert [step["step"] for step in STATIC_STEPS] == list(range(1, 12))
    assert STATIC_STEPS[0]["paper_tool"] == "TSPipline"
    assert "search_transition_state_path" in STATIC_STEPS[0]["candidate_actions"]
    assert STATIC_STEPS[3]["paper_tool"] == "Gaussian"
    assert STATIC_STEPS[6]["paper_tool"] == "Gaussian"
    assert STATIC_STEPS[9]["paper_tool"] == "Gaussian"
    assert STATIC_STEPS[-1]["paper_tool"] == "Python script"


def test_case1_static_mapping_uses_author_gaussian_preparation_actions() -> None:
    """论文 Step 2/3、5/6、8/9 应映射到正式目录中的作者工具。"""

    expected = {
        2: "generate_gaussian_code",
        3: "xyz_to_gjf",
        5: "generate_gaussian_code",
        6: "xyz_to_gjf",
        8: "generate_gaussian_code",
        9: "xyz_to_gjf",
    }
    steps = {item["step"]: item for item in STATIC_STEPS}
    for step_number, action_id in expected.items():
        assert action_id in steps[step_number]["candidate_actions"]
        assert "catalog 已提供作者原名 Action" in steps[step_number]["gap"]


def test_transition_state_path_example_exposes_only_public_inputs() -> None:
    """示例请求只暴露反应物/产物和已注册力场，隐藏控制仍不可覆盖。"""

    example = Path(__file__).parents[1] / "third_party" / "minichem_toolbox" / "examples" / "transition_state_path_request.json"
    request = json.loads(example.read_text(encoding="utf-8"))
    assert request["action_id"] == "search_transition_state_path"
    assert request["backend_id"] == "transition_state_path_search"
    assert request["inputs"]["reactants"]
    assert request["inputs"]["products"]
    assert request["inputs"]["main_force_field"] == "MMFF"
    assert request["inputs"]["target_force_field"] == "UMA"
    assert request["method_spec"] == {}
    assert request["action_settings"] == {}
    assert request["resource_limits"]["gpu_count"] == 1


def test_case1_reference_results_preserve_method_and_state_facts() -> None:
    """case1.json 的已解析结果应保留方法、电子态、坐标和虚频事实。"""

    summary = load_case1_summary()
    assert summary["sections"]["opt"]["record_count"] == 14
    assert summary["sections"]["sp"]["record_count"] == 14
    opt = summary["sections"]["opt"]["records"]["TS-2h-1"]
    assert opt["functional"] == "B3LYP"
    assert opt["charge"] == 1
    assert opt["multiplicity"] == 1
    assert opt["natom"] == 53
    assert opt["temperature"] == 303.15
    assert opt["imaginary_frequency_count"] >= 1
    assert opt["has_coordinates"] is True
    sp = summary["sections"]["sp"]["records"]["TS-2h-1"]
    assert sp["functional"] == "M062X"
    assert sp["keywords"]


def test_case1_step11_uses_ev_to_hartree_conversion() -> None:
    """公开数值回放应得到约 1.88485 kcal/mol 和 91.6% ee。"""

    data = json.load(open("supplementary/case1.json", encoding="utf-8"))
    result = calculate_case1_delta_g_from_published_values(data)
    assert result["units"]["freeenergy"] == "Hartree"
    assert result["units"]["scfenergy_final"] == "eV"
    assert result["delta_delta_g_kcal_mol"] == pytest.approx(1.8848503, abs=1e-6)
    assert result["predicted_ee_percent"] == pytest.approx(91.61298, abs=1e-4)


def test_case1_native_decks_preserve_exact_paper_routes() -> None:
    """native deck builder 必须保留附录要求的 TS、IRC 和单点 route。"""

    record = json.loads(open("supplementary/case1.json", encoding="utf-8").read())["opt"]["TS-2a"]
    routes = {plan["stage"]: plan["route"] for plan in CASE1_NATIVE_REPLAY_PLAN}
    opt_deck = build_case1_gaussian_deck(
        record,
        route=routes["step4_opt_freq"],
        title="Case 1 TS-2a",
        checkpoint_name="ts_2a.chk",
    )
    irc_deck = build_case1_gaussian_deck(
        record,
        route=routes["step7_irc"],
        title="Case 1 TS-2a IRC",
        checkpoint_name="ts_2a_irc.chk",
    )
    sp_deck = build_case1_gaussian_deck(
        record,
        route=routes["step10_single_point"],
        title="Case 1 TS-2a SP",
        checkpoint_name="ts_2a_sp.chk",
    )
    assert "Opt=(TS,NoEigenTest,CalcAll,MaxStep=5)" in opt_deck
    assert "Temperature=303.15" in opt_deck
    assert "IRC=(CalcFC,LQA,MaxPoints=50,ReCalc=5)" in irc_deck
    assert "M062X/Def2TZVPP SP" in sp_deck
    assert "Int=UltraFine SCF=Tight" in sp_deck
    assert opt_deck.startswith("%NProcShared=60\n%Mem=320000MB\n")
    assert opt_deck.endswith("\n\n")
    assert opt_deck.count("\n") == 62


def test_case1_downstream_jobs_fit_fixed_cpu_budget() -> None:
    """每个 native Gaussian 作业都独占 60 核、320000 MB 和 0 GPU。"""

    expected = {
        "cpu_cores": CASE1_NATIVE_CPU_CORES,
        "memory_mb": CASE1_NATIVE_MEMORY_MB,
        "gpu_count": CASE1_NATIVE_GPU_COUNT,
    }
    assert expected == {
        "cpu_cores": 60,
        "memory_mb": 320000,
        "gpu_count": 0,
    }
    for plan in CASE1_NATIVE_REPLAY_PLAN:
        assert {
            "cpu_cores": plan["cpu_cores"],
            "memory_mb": plan["memory_mb"],
            "gpu_count": plan["gpu_count"],
        } == expected
    # 旧 8/16 核计划只能作为历史证据，不能由兼容 helper 传播到新提交。
    assert case1_native_resource_limits({"cpu_cores": 8, "memory_mb": 32768}) == expected


def test_case1_native_request_and_operation_identity_are_stable() -> None:
    """native 请求不携带伪造 job 身份，operation_id 由脚本固定生成。"""

    request = build_case1_native_request(
        deck_path="code/case1_native/ts_2a_step4_opt_freq.gjf",
        label="ARCHE Case 1 TS-2a step4_opt_freq",
        calculation_intent="transition_state",
    )
    assert request["software_id"] == "gaussian"
    assert request["executable"] == "g16"
    assert request["stdin_target"] == "input.gjf"
    assert request["resource_limits"] == {
        "cpu_cores": CASE1_NATIVE_CPU_CORES,
        "memory_mb": CASE1_NATIVE_MEMORY_MB,
        "gpu_count": CASE1_NATIVE_GPU_COUNT,
    }
    assert "operation_id" not in request
    assert "job_id" not in request
    assert build_native_operation_id("TS-2a", "step4_opt_freq") == "case1-ts_2a-step4-opt-freq"
    assert build_native_operation_id(
        "TS-2a", "step4_opt_freq", attempt_index=2
    ) == "case1-ts_2a-step4-opt-freq-attempt-2"


def test_case1_native_scheduler_submits_only_first_operation() -> None:
    """前一 Gaussian 等待时，后续 operation 只记录 serial blocked。"""

    record = json.loads(open("supplementary/case1.json", encoding="utf-8").read())["opt"]["TS-2a"]

    class SerialAdapter:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def ensure_workspace_text(self, _path: str, _content: str) -> dict:
            self.calls.append(("ensure", _path))
            return {"status": "success"}

        def validate_native_job(self, _request: dict, operation_id: str) -> dict:
            self.calls.append(("validate", operation_id))
            return {"status": "success", "valid": True}

        def submit_native_job(self, request: dict, operation_id: str) -> dict:
            assert request["resource_limits"] == {
                "cpu_cores": 60,
                "memory_mb": 320000,
                "gpu_count": 0,
            }
            self.calls.append(("submit", operation_id))
            return {"status": "waiting", "job_id": "job_serial_first"}

    adapter = SerialAdapter()
    runs: dict[str, dict] = {}
    first = run_case1_native_step(
        adapter=adapter,
        runs=runs,
        save_manifest=lambda: None,
        label="TS-2a",
        record=record,
        plan=CASE1_NATIVE_REPLAY_PLAN[0],
        gaussian_available=True,
        operation_root=Path("/tmp/case1-serial-test"),
    )
    blocked = run_case1_native_step(
        adapter=adapter,
        runs=runs,
        save_manifest=lambda: None,
        label="TS-2b-1",
        record=record,
        plan=CASE1_NATIVE_REPLAY_PLAN[0],
        gaussian_available=True,
        operation_root=Path("/tmp/case1-serial-test"),
    )

    assert first["status"] == "waiting"
    assert blocked["status"] == "serial_blocked"
    assert blocked["submission_attempted"] is False
    assert blocked["serial_blocked_by_operation_id"] == first["operation_id"]
    assert blocked["operation_id"] == "case1-ts_2b_1-step4-opt-freq"
    assert [kind for kind, _value in adapter.calls] == ["ensure", "validate", "submit"]


def test_case1_serial_block_recovers_after_previous_operation_is_collected(
    tmp_path: Path,
) -> None:
    """前一 operation 收集终态后，下一次运行自动解锁原 operation_id。"""

    record = json.loads(open("supplementary/case1.json", encoding="utf-8").read())["opt"]["TS-2a"]

    class RecoveryAdapter:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def ensure_workspace_text(self, _path: str, _content: str) -> dict:
            self.calls.append(("ensure", _path))
            return {"status": "success"}

        def validate_native_job(self, _request: dict, operation_id: str) -> dict:
            self.calls.append(("validate", operation_id))
            return {"status": "success", "valid": True}

        def submit_native_job(self, _request: dict, operation_id: str) -> dict:
            self.calls.append(("submit", operation_id))
            return {"status": "waiting", "job_id": f"job_{len(self.calls)}"}

        def resume_operation(self, operation_id: str, *, collect: bool = True) -> dict:
            assert collect is True
            self.calls.append(("resume", operation_id))
            return {
                "status": "completed",
                "ready": True,
                "job_id": "job_serial_first",
                "software_status": "normal_termination",
                "convergence_status": "converged",
                "scientific_validation_status": "mechanically_valid",
                "outputs": [
                    {"path": "outputs/serial_first/stdout.log"},
                    {"path": "outputs/serial_first/step4.chk"},
                ],
            }

    adapter = RecoveryAdapter()
    runs: dict[str, dict] = {}
    run_case1_native_step(
        adapter=adapter,
        runs=runs,
        save_manifest=lambda: None,
        label="TS-2a",
        record=record,
        plan=CASE1_NATIVE_REPLAY_PLAN[0],
        gaussian_available=True,
        operation_root=tmp_path,
    )
    blocked = run_case1_native_step(
        adapter=adapter,
        runs=runs,
        save_manifest=lambda: None,
        label="TS-2b-1",
        record=record,
        plan=CASE1_NATIVE_REPLAY_PLAN[0],
        gaussian_available=True,
        operation_root=tmp_path,
    )
    first_operation_id = blocked["operation_id"]

    resumed = run_case1_native_step(
        adapter=adapter,
        runs=runs,
        save_manifest=lambda: None,
        label="TS-2a",
        record=record,
        plan=CASE1_NATIVE_REPLAY_PLAN[0],
        gaussian_available=True,
        operation_root=tmp_path,
    )
    unlocked = run_case1_native_step(
        adapter=adapter,
        runs=runs,
        save_manifest=lambda: None,
        label="TS-2b-1",
        record=record,
        plan=CASE1_NATIVE_REPLAY_PLAN[0],
        gaussian_available=True,
        operation_root=tmp_path,
    )

    assert resumed["status"] == "completed"
    assert unlocked["status"] == "waiting"
    assert unlocked["operation_id"] == first_operation_id
    assert [kind for kind, _value in adapter.calls].count("submit") == 2
    assert [kind for kind, _value in adapter.calls].count("resume") == 1


def test_cancelled_attempt_is_archived_when_serial_retry_unlocks(tmp_path: Path) -> None:
    """取消的 attempt 在串行阻塞解除后归档，并只提交 attempt-3。"""

    record = json.loads(open("supplementary/case1.json", encoding="utf-8").read())["opt"]["TS-2a"]
    parent = {
        "status": "completed",
        "ready": True,
        "operation_id": "operation-parent-serial",
        "job_id": "job_parent_serial",
        "software_status": "normal_termination",
        "convergence_status": "converged",
        "scientific_validation_status": "mechanically_valid",
        "outputs": [{"path": "outputs/parent_serial/step4.chk"}],
    }

    class RetryAdapter:
        def __init__(self) -> None:
            self.submitted: list[str] = []

        def ensure_workspace_text(self, _path: str, _content: str) -> dict:
            return {"status": "success"}

        def validate_native_job(self, _request: dict, operation_id: str) -> dict:
            self.submitted.append(f"validate:{operation_id}")
            return {"status": "success", "valid": True}

        def submit_native_job(self, request: dict, operation_id: str) -> dict:
            assert request["resource_limits"] == {
                "cpu_cores": 60,
                "memory_mb": 320000,
                "gpu_count": 0,
            }
            self.submitted.append(f"submit:{operation_id}")
            return {"status": "waiting", "job_id": "job_retry_attempt_3"}

    adapter = RetryAdapter()
    runs: dict[str, dict] = {
        "native_TS-2a_step4_opt_freq": {
            "status": "waiting",
            "operation_id": "case1-ts_2a-step4-opt-freq",
            "submission_attempted": True,
        },
        "native_TS-2b-1_step7_irc": {
            "status": "technical_pause",
            "minichem_status": "cancelled",
            "job_id": "job_cancelled_attempt_2",
            "operation_id": "case1-ts_2b_1-step7-irc-attempt-2",
            "attempt_index": 2,
            "submission_attempted": True,
        },
    }
    blocked = run_case1_native_step(
        adapter=adapter,
        runs=runs,
        save_manifest=lambda: None,
        label="TS-2b-1",
        record=record,
        plan=CASE1_NATIVE_REPLAY_PLAN[1],
        gaussian_available=True,
        operation_root=tmp_path,
        dependency_result=parent,
        retry_failed=True,
    )
    assert blocked["status"] == "serial_blocked"
    assert "native_TS-2b-1_step7_irc_attempt_2" not in runs
    assert adapter.submitted == []

    runs["native_TS-2a_step4_opt_freq"] = parent
    retried = run_case1_native_step(
        adapter=adapter,
        runs=runs,
        save_manifest=lambda: None,
        label="TS-2b-1",
        record=record,
        plan=CASE1_NATIVE_REPLAY_PLAN[1],
        gaussian_available=True,
        operation_root=tmp_path,
        dependency_result=parent,
        retry_failed=True,
    )
    assert runs["native_TS-2b-1_step7_irc_attempt_2"]["job_id"] == (
        "job_cancelled_attempt_2"
    )
    assert retried["operation_id"] == "case1-ts_2b_1-step7-irc-attempt-3"
    assert retried["attempt_index"] == 3
    assert adapter.submitted == [
        "validate:case1-ts_2b_1-step7-irc-attempt-3",
        "submit:case1-ts_2b_1-step7-irc-attempt-3",
    ]


def test_native_output_paths_restore_promoted_manifest_paths() -> None:
    """旧 manifest 只有顶层摘要路径时，Step 11 仍可接收 native 日志。"""

    value = {
        "status": "completed",
        "log_path": "outputs/jobs/job_a/stdout.log",
        "checkpoint_path": "outputs/jobs/job_a/ts.chk",
    }

    assert extract_native_log_path(value) == "outputs/jobs/job_a/stdout.log"
    assert extract_native_checkpoint_path(value) == "outputs/jobs/job_a/ts.chk"


def test_terminal_native_failure_requires_explicit_new_attempt(tmp_path: Path) -> None:
    """明确失败默认保持不动；显式重试时归档旧 job 并使用新 operation。"""

    record = json.loads(open("supplementary/case1.json", encoding="utf-8").read())["opt"]["TS-2a"]

    class NativeAdapter:
        def __init__(self) -> None:
            self.operations: list[str] = []

        def ensure_workspace_text(self, _path: str, _content: str) -> dict:
            return {"status": "success"}

        def validate_native_job(self, _request: dict, operation_id: str) -> dict:
            self.operations.append(f"validate:{operation_id}")
            return {"status": "success", "valid": True}

        def submit_native_job(self, _request: dict, operation_id: str) -> dict:
            self.operations.append(f"submit:{operation_id}")
            return {
                "status": "waiting",
                "job_id": "job_retry_0000000000000000000000000001",
            }

    failed = {
        "status": "technical_pause",
        "minichem_status": "failed",
        "job_id": "job_failed_000000000000000000000000001",
        "operation_id": "case1-ts_2a-step4-opt-freq",
        "attempt_index": 1,
    }
    adapter = NativeAdapter()
    runs = {"native_TS-2a_step4_opt_freq": dict(failed)}
    unchanged = run_case1_native_step(
        adapter=adapter,
        runs=runs,
        save_manifest=lambda: None,
        label="TS-2a",
        record=record,
        plan=CASE1_NATIVE_REPLAY_PLAN[0],
        gaussian_available=True,
        operation_root=tmp_path,
    )
    assert unchanged == failed
    assert adapter.operations == []

    retried = run_case1_native_step(
        adapter=adapter,
        runs=runs,
        save_manifest=lambda: None,
        label="TS-2a",
        record=record,
        plan=CASE1_NATIVE_REPLAY_PLAN[0],
        gaussian_available=True,
        operation_root=tmp_path,
        retry_failed=True,
    )
    new_operation = "case1-ts_2a-step4-opt-freq-attempt-2"
    assert runs["native_TS-2a_step4_opt_freq_attempt_1"]["job_id"] == failed["job_id"]
    assert retried["operation_id"] == new_operation
    assert retried["attempt_index"] == 2
    assert retried["deck_path"].endswith("step4_opt_freq_attempt_2.gjf")
    assert adapter.operations == [
        f"validate:{new_operation}",
        f"submit:{new_operation}",
    ]


def test_existing_native_attempt_is_never_resubmitted() -> None:
    """已有未决 native attempt 必须暂停给人工恢复，不能重复提交副作用。"""

    record = json.loads(open("supplementary/case1.json", encoding="utf-8").read())["opt"]["TS-2a"]
    calls: list[str] = []

    class FakeAdapter:
        def __getattr__(self, name: str):
            calls.append(name)
            raise AssertionError(f"不应调用 MCP 方法：{name}")

    runs = {"native_TS-2a_step4_opt_freq": {"status": "in_progress"}}
    result = run_case1_native_step(
        adapter=FakeAdapter(),
        runs=runs,
        save_manifest=lambda: None,
        label="TS-2a",
        record=record,
        plan=CASE1_NATIVE_REPLAY_PLAN[0],
        gaussian_available=True,
        operation_root=Path("/tmp/case1-operations"),
    )
    assert result["status"] == "uncertain_requires_manual_recovery"
    assert result["recovery_required"] is True
    assert calls == []


def test_existing_action_and_jobs_only_resume_without_new_submit(tmp_path: Path) -> None:
    """第二次运行不增加 execute_action、native submit 或 analysis submit 次数。"""

    class CountingAdapter:
        def __init__(self) -> None:
            self.execute_calls = 0
            self.native_submit_calls = 0
            self.analysis_submit_calls = 0
            self.resume_calls: list[str] = []

        def execute_action(self, _request):
            self.execute_calls += 1
            raise AssertionError("已有 Action 不应再次执行")

        def submit_native_job(self, _request, _operation_id):
            self.native_submit_calls += 1
            raise AssertionError("已有 native job 不应再次提交")

        def submit_analysis_program(self, _request, _operation_id):
            self.analysis_submit_calls += 1
            raise AssertionError("已有 analysis job 不应再次提交")

        def resume_operation(self, operation_id, *, collect=True):
            assert collect is True
            self.resume_calls.append(operation_id)
            return {"status": "completed", "job_id": "job_00000000000000000000000000000001"}

    adapter = CountingAdapter()
    runs = {
        "standardize_reactant": {"status": "completed"},
        "native_TS-2a_step4_opt_freq": {
            "status": "waiting",
            "operation_id": "case1-ts_2a-step4-opt-freq",
        },
        "step11_python_delta_g_analysis": {
            "status": "waiting",
            "operation_id": "case1-step11-delta-g-analysis",
        },
    }
    run_mcp_action_once(
        adapter=adapter,
        runs=runs,
        key="standardize_reactant",
        request={"action_id": "standardize_structure", "backend_id": "rdkit"},
        save_manifest=lambda: None,
    )
    (tmp_path / "case1-ts_2a-step4-opt-freq.json").write_text("{}", encoding="utf-8")
    (tmp_path / "case1-step11-delta-g-analysis.json").write_text("{}", encoding="utf-8")
    recovered_native = run_case1_native_step(
        adapter=adapter,
        runs=runs,
        save_manifest=lambda: None,
        label="TS-2a",
        record=json.load(open("supplementary/case1.json", encoding="utf-8"))["opt"]["TS-2a"],
        plan=CASE1_NATIVE_REPLAY_PLAN[0],
        gaussian_available=True,
        operation_root=tmp_path,
    )
    recovered_analysis = recover_existing_program_operation(
        adapter=adapter,
        runs=runs,
        key="step11_python_delta_g_analysis",
        save_manifest=lambda: None,
        operation_root=tmp_path,
    )
    assert recovered_native["recovered_without_resubmission"] is True
    assert recovered_analysis["recovered_without_resubmission"] is True
    assert adapter.execute_calls == 0
    assert adapter.native_submit_calls == 0
    assert adapter.analysis_submit_calls == 0
    assert adapter.resume_calls == [
        "case1-ts_2a-step4-opt-freq",
        "case1-step11-delta-g-analysis",
    ]


def test_native_recovery_query_failure_can_retry_same_receipt(tmp_path: Path) -> None:
    """只读恢复查询失败后应继续查询同一 operation，不能重提 native 作业。"""

    operation_id = "case1-ts_2a-step4-opt-freq"
    (tmp_path / f"{operation_id}.json").write_text("{}", encoding="utf-8")

    class RecoveryAdapter:
        def __init__(self) -> None:
            self.resume_calls: list[str] = []
            self.submit_calls = 0

        def resume_operation(self, received_operation_id: str, *, collect: bool = True):
            assert collect is True
            self.resume_calls.append(received_operation_id)
            return {
                "status": "waiting",
                "job_id": "job_00000000000000000000000000000001",
            }

        def submit_native_job(self, _request: dict, _operation_id: str):
            self.submit_calls += 1
            raise AssertionError("恢复查询不得重新提交 native 作业")

    adapter = RecoveryAdapter()
    runs = {
        "native_TS-2a_step4_opt_freq": {
            "status": "recovery_failed",
            "submission_attempted": True,
            "operation_id": operation_id,
            "request_fingerprint": "a" * 64,
            "job_id": "job_00000000000000000000000000000001",
            "error": "上一次查询响应包含错误的旁路 job_id",
        }
    }
    result = run_case1_native_step(
        adapter=adapter,
        runs=runs,
        save_manifest=lambda: None,
        label="TS-2a",
        record=json.load(open("supplementary/case1.json", encoding="utf-8"))["opt"]["TS-2a"],
        plan=CASE1_NATIVE_REPLAY_PLAN[0],
        gaussian_available=True,
        operation_root=tmp_path,
    )

    assert result["status"] == "waiting"
    assert result["recovered_without_resubmission"] is True
    assert adapter.resume_calls == [operation_id]
    assert adapter.submit_calls == 0


def test_downstream_native_step_is_blocked_until_step4_checkpoint_is_terminal() -> None:
    """Step 7/10 在 Step 4 未完成时不得写 deck 或提交副作用。"""

    record = json.loads(open("supplementary/case1.json", encoding="utf-8").read())["opt"]["TS-2a"]
    calls: list[str] = []

    class FakeAdapter:
        def __getattr__(self, name: str):
            calls.append(name)
            raise AssertionError(f"依赖未完成时不应调用 MCP 方法：{name}")

    runs: dict[str, dict] = {}
    result = run_case1_native_step(
        adapter=FakeAdapter(),
        runs=runs,
        save_manifest=lambda: None,
        label="TS-2a",
        record=record,
        plan=CASE1_NATIVE_REPLAY_PLAN[1],
        gaussian_available=True,
        operation_root=Path("/tmp/case1-operations"),
        dependency_result={"status": "waiting", "job_id": "job_parent"},
    )
    assert result["status"] == "blocked_dependency"
    assert result["submission_attempted"] is False
    assert calls == []


def test_downstream_native_step_stages_step4_checkpoint_and_parent_job(tmp_path: Path) -> None:
    """终态 Step 4 的实际 checkpoint 必须成为 Step 7 的 staged input。"""

    record = json.loads(open("supplementary/case1.json", encoding="utf-8").read())["opt"]["TS-2a"]

    class NativeAdapter:
        def __init__(self) -> None:
            self.request = None
            self.deck = ""
            self.submit_calls = 0

        def ensure_workspace_text(self, _path: str, content: str) -> dict:
            self.deck = content
            return {"status": "success", "reused": False}

        def validate_native_job(self, _request: dict, _operation_id: str) -> dict:
            return {"status": "success", "valid": True}

        def submit_native_job(self, request: dict, _operation_id: str) -> dict:
            self.submit_calls += 1
            self.request = request
            return {
                "status": "completed",
                "job_id": "job_child_00000000000000000000000000000001",
                "ready": True,
                "outputs": [
                    {
                        "path": "outputs/execution_jobs/job_child_00000000000000000000000000000001/stdout.log",
                        "job_relative_path": "stdout.log",
                    }
                ],
            }

    adapter = NativeAdapter()
    parent_checkpoint = "outputs/execution_jobs/job_parent_00000000000000000000000000000001/ts_2a_step4_opt_freq.chk"
    parent = {
        "status": "completed",
        "ready": True,
        "operation_id": "operation-parent-serial",
        "job_id": "job_parent_00000000000000000000000000000001",
        "software_status": "normal_termination",
        "convergence_status": "converged",
        "scientific_validation_status": "mechanically_valid",
        "outputs": [{"path": parent_checkpoint, "job_relative_path": "ts_2a_step4_opt_freq.chk"}],
    }
    runs: dict[str, dict] = {}
    result = run_case1_native_step(
        adapter=adapter,
        runs=runs,
        save_manifest=lambda: None,
        label="TS-2a",
        record=record,
        plan=CASE1_NATIVE_REPLAY_PLAN[1],
        gaussian_available=True,
        operation_root=tmp_path,
        dependency_result=parent,
    )
    assert result["status"] == "completed"
    assert result["parent_job_id"] == parent["job_id"]
    assert result["parent_checkpoint_path"] == parent_checkpoint
    assert extract_native_checkpoint_path(parent) == parent_checkpoint
    assert adapter.submit_calls == 1
    assert adapter.request is not None
    assert adapter.request["parent_job_id"] == parent["job_id"]
    assert adapter.request["staged_inputs"][1] == {
        "source_path": parent_checkpoint,
        "target_path": "ts_2a_step4_opt_freq.chk",
    }
    assert "%OldChk=ts_2a_step4_opt_freq.chk" in adapter.deck
    assert "Geom=AllCheck" in adapter.deck
    assert "Guess=Read" in adapter.deck
    assert "53" not in adapter.deck


def test_downstream_native_step_rejects_terminal_but_invalid_step4(
    tmp_path: Path,
) -> None:
    """进程结束但 TS 判据失败的 Step 4 不能解锁 IRC 或单点计算。"""

    record = json.loads(open("supplementary/case1.json", encoding="utf-8").read())["opt"]["TS-2a"]
    calls: list[str] = []

    class FakeAdapter:
        def __getattr__(self, name: str):
            calls.append(name)
            raise AssertionError(f"无效上游不应调用 MCP 方法：{name}")

    parent = {
        "status": "completed",
        "ready": True,
        "operation_id": "operation-parent-chain",
        "job_id": "job_parent_00000000000000000000000000000001",
        "software_status": "normal_termination",
        "convergence_status": "failed",
        "scientific_validation_status": "mechanically_invalid",
        "outputs": [{"path": "outputs/jobs/parent.chk"}],
    }
    result = run_case1_native_step(
        adapter=FakeAdapter(),
        runs={},
        save_manifest=lambda: None,
        label="TS-2a",
        record=record,
        plan=CASE1_NATIVE_REPLAY_PLAN[1],
        gaussian_available=True,
        operation_root=tmp_path,
        dependency_result=parent,
    )

    assert check_native_job_scientific_gate(parent) is False
    assert result["status"] == "blocked_dependency"
    assert "机械科学有效性" in result["blocked_reason"]
    assert calls == []


def test_native_chain_recovery_does_not_submit_downstream_twice(tmp_path: Path) -> None:
    """下游已有 operation 时再次运行只能 resume 同一个 operation。"""

    record = json.loads(open("supplementary/case1.json", encoding="utf-8").read())["opt"]["TS-2a"]
    parent = {
        "status": "completed",
        "ready": True,
        "operation_id": "operation-parent-chain",
        "job_id": "job_parent_00000000000000000000000000000001",
        "software_status": "normal_termination",
        "convergence_status": "converged",
        "scientific_validation_status": "mechanically_valid",
        "outputs": [{"path": "outputs/execution_jobs/job_parent_00000000000000000000000000000001/parent.chk"}],
    }

    class NativeAdapter:
        def __init__(self) -> None:
            self.submit_calls = 0
            self.resume_calls = 0

        def ensure_workspace_text(self, _path: str, _content: str) -> dict:
            return {"status": "success"}

        def validate_native_job(self, _request: dict, _operation_id: str) -> dict:
            return {"status": "success", "valid": True}

        def submit_native_job(self, _request: dict, _operation_id: str) -> dict:
            self.submit_calls += 1
            return {"status": "waiting", "job_id": "job_child_00000000000000000000000000000001"}

        def resume_operation(self, _operation_id: str, *, collect: bool = True) -> dict:
            assert collect is True
            self.resume_calls += 1
            return {"status": "waiting", "job_id": "job_child_00000000000000000000000000000001"}

    adapter = NativeAdapter()
    runs: dict[str, dict] = {}
    first = run_case1_native_step(
        adapter=adapter,
        runs=runs,
        save_manifest=lambda: None,
        label="TS-2a",
        record=record,
        plan=CASE1_NATIVE_REPLAY_PLAN[1],
        gaussian_available=True,
        operation_root=tmp_path,
        dependency_result=parent,
    )
    assert first["status"] == "waiting"
    second = run_case1_native_step(
        adapter=adapter,
        runs=runs,
        save_manifest=lambda: None,
        label="TS-2a",
        record=record,
        plan=CASE1_NATIVE_REPLAY_PLAN[1],
        gaussian_available=True,
        operation_root=tmp_path,
        dependency_result=parent,
    )
    assert second["recovered_without_resubmission"] is True
    assert adapter.submit_calls == 1
    assert adapter.resume_calls == 1


def test_step11_analysis_declares_only_fresh_native_logs() -> None:
    """Step 11 只声明附录要求的 Step 4/10 日志，不携带参考数值。"""

    input_paths = {
        "ts_2a_step4_opt_freq": "outputs/execution_jobs/job_a/step4/stdout.log",
        "ts_2a_step10_single_point": "outputs/execution_jobs/job_a/step10/stdout.log",
        "ts_2b_1_step4_opt_freq": "outputs/execution_jobs/job_b/step4/stdout.log",
        "ts_2b_1_step10_single_point": "outputs/execution_jobs/job_b/step10/stdout.log",
    }
    program = build_case1_fresh_analysis_program(sorted(input_paths))
    request = build_case1_fresh_analysis_request(input_paths)
    compile(program, "<case1-fresh-analysis>", "exec")
    assert "case1.json" not in program
    assert "case1_values" not in program
    assert "SCF Done" in program
    assert "Sum of electronic and thermal Free Energies" in program
    assert "Normal termination of Gaussian" in program
    assert "Optimization completed" in program
    assert "NImag=" in program
    assert "path complete" not in program
    assert "irc_bidirectional_completed" not in program
    assert {item["name"] for item in request["inputs"]} == set(input_paths)
    assert {item["source_path"] for item in request["inputs"]} == set(input_paths.values())


def _run_case1_analysis_program(
    program: str,
    logs: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    """用内存 JobContext 执行生成程序，返回它写出的 JSON 结果。"""

    written: dict[str, dict] = {}

    class FakeInput:
        def __init__(self, text: str) -> None:
            self.text = text

        def read_text(self, *, encoding: str) -> str:
            assert encoding == "utf-8"
            return self.text

    class FakeJobContext:
        @classmethod
        def load(cls):
            return cls()

        def input(self, name: str) -> FakeInput:
            return FakeInput(logs[name])

        def write_json(self, name: str, value: dict) -> None:
            written[name] = value

    module = types.ModuleType("researchchem_job")
    module.JobContext = FakeJobContext
    monkeypatch.setitem(sys.modules, "researchchem_job", module)
    exec(compile(program, "<case1-fresh-analysis>", "exec"), {})
    return written["delta_g_result"]


def test_step11_computes_delta_g_without_irc_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """附录 Step 11 只使用 Step 4 热化学和 Step 10 电子能。"""

    inputs = {
        f"ts_2{suffix}_{stage}": f"{suffix}/{stage}.log"
        for suffix in ("a", "b_1")
        for stage in ("step4_opt_freq", "step10_single_point")
    }
    opt_log = (
        "SCF Done:  E(RB3LYP) = -100.000000 A.U.\n"
        "Sum of electronic and thermal Free Energies= -99.000000\n"
        "Optimization completed\nNImag= 1\nNormal termination of Gaussian\n"
    )
    sp_log = "SCF Done:  E(RM062X) = -101.000000 A.U.\nNormal termination of Gaussian\n"
    logs = {}
    for name in inputs:
        if name.endswith("step4_opt_freq"):
            logs[name] = opt_log
        else:
            logs[name] = sp_log
    program = build_case1_fresh_analysis_program(sorted(inputs))

    completed = _run_case1_analysis_program(program, logs, monkeypatch)
    assert completed["status"] == "completed"
    assert completed["errors"] == []
    assert completed["delta_delta_g_kcal_mol"] == pytest.approx(0.0)


def test_step11_parses_standard_gaussian_frequency_lines_and_fortran_exponents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """标准 Gaussian 日志没有 NImag 标记时仍应从 Frequencies 行识别虚频。"""

    inputs = {
        f"ts_2{suffix}_{stage}": f"{suffix}/{stage}.log"
        for suffix in ("a", "b_1")
        for stage in ("step4_opt_freq", "step10_single_point")
    }
    opt_log = (
        " SCF Done:  E(RB3LYP) =  -1.00000000D+02 A.U.\n"
        " Sum of electronic and thermal Free Energies= -9.90000000D+01\n"
        " Optimization completed\n"
        " Frequencies --  -1.234D+02   2.000D+01   3.000D+01\n"
        " Normal termination of Gaussian 16\n"
    )
    sp_log = (
        " SCF Done:  E(RM062X) =  -1.01000000D+02 A.U.\n"
        " Normal termination of Gaussian 16\n"
    )
    logs = {
        name: opt_log if name.endswith("step4_opt_freq") else sp_log
        for name in inputs
    }

    result = _run_case1_analysis_program(
        build_case1_fresh_analysis_program(sorted(inputs)),
        logs,
        monkeypatch,
    )

    assert result["status"] == "completed"
    assert result["errors"] == []
    assert result["parsed_native_records"]["TS-2a"]["step4_opt_freq"][
        "imaginary_frequency_count"
    ] == 1
    assert result["delta_delta_g_kcal_mol"] == pytest.approx(0.0)


def test_step11_analysis_preflight_failure_does_not_submit_program(tmp_path: Path) -> None:
    """分析程序 preflight 失败时只保存验证证据，不产生提交副作用。"""

    runs = {}
    for label in ("TS-2a", "TS-2b-1"):
        step4_key = f"native_{label}_step4_opt_freq"
        runs[step4_key] = _decorate_valid_native_record(
            tmp_path,
            step4_key,
            intent="transition_state",
        )
        stage = "step10_single_point"
        key = f"native_{label}_{stage}"
        runs[key] = _decorate_valid_native_record(
            tmp_path,
            key,
            intent="single_point",
            parent=runs[step4_key],
        )

    class ValidationAdapter:
        def __init__(self) -> None:
            self.validate_calls = 0
            self.submit_calls = 0

        def ensure_workspace_text(self, _path: str, _content: str) -> dict:
            return {"status": "success"}

        def validate_analysis_program(self, _request: dict, _operation_id: str) -> dict:
            self.validate_calls += 1
            return {"status": "success", "valid": False, "error": "invalid output contract"}

        def submit_analysis_program(self, _request: dict, _operation_id: str) -> dict:
            self.submit_calls += 1
            raise AssertionError("preflight 失败时不应提交 analysis program")

    adapter = ValidationAdapter()
    result = run_case1_fresh_analysis(
        adapter=adapter,
        runs=runs,
        save_manifest=lambda: None,
        operation_root=tmp_path,
    )

    assert result["status"] == "analysis_validation_failed"
    assert result["submission_attempted"] is False
    assert adapter.validate_calls == 1
    assert adapter.submit_calls == 0


def test_step11_manifest_distinguishes_program_completion_from_scientific_result() -> None:
    """分析程序完成时，manifest 仍须传播内部科学证据状态。"""

    class FakeAdapter:
        def __init__(self, payload: dict) -> None:
            self.payload = payload

        def read_workspace_text(self, path: str) -> dict:
            assert path.endswith("case1_fresh_delta_g_result.json")
            return {"content": json.dumps(self.payload)}

    technical_result = {
        "status": "completed",
        "outputs": [
            {"path": "outputs/case1_fresh_delta_g_result.json"},
        ],
    }
    incomplete = attach_case1_analysis_outcome(
        FakeAdapter(
            {
                "status": "insufficient_fresh_evidence",
                "delta_delta_g_kcal_mol": None,
                "errors": ["missing bidirectional Step 7 IRC evidence: TS-2b-1"],
            }
        ),
        dict(technical_result),
        1.9,
    )
    assert incomplete["status"] == "completed"
    assert incomplete["case1_analysis_status"] == "insufficient_fresh_evidence"
    assert incomplete["case1_scientifically_valid"] is False
    assert "paper_comparison" not in incomplete

    completed = attach_case1_analysis_outcome(
        FakeAdapter(
            {
                "status": "completed",
                "delta_delta_g_kcal_mol": 1.8,
                "errors": [],
            }
        ),
        dict(technical_result),
        1.9,
    )
    assert completed["case1_analysis_status"] == "completed"
    assert completed["case1_scientifically_valid"] is True
    assert completed["paper_comparison"]["absolute_difference_kcal_mol"] == pytest.approx(0.1)


def test_step11_waits_for_all_numeric_dependency_logs() -> None:
    """Step 4/10 任一仍 waiting 时，分析输入收集必须返回 None。"""

    runs = {
        "native_TS-2a_step4_opt_freq": {"status": "completed", "outputs": [{"path": "a/step4/stdout.log"}]},
        "native_TS-2a_step7_irc": {"status": "completed", "outputs": [{"path": "a/step7/stdout.log"}]},
        "native_TS-2a_step10_single_point": {"status": "waiting", "outputs": []},
        "native_TS-2b-1_step4_opt_freq": {"status": "completed", "outputs": [{"path": "b/step4/stdout.log"}]},
        "native_TS-2b-1_step7_irc": {"status": "completed", "outputs": [{"path": "b/step7/stdout.log"}]},
        "native_TS-2b-1_step10_single_point": {"status": "completed", "outputs": [{"path": "b/step10/stdout.log"}]},
    }
    assert collect_case1_fresh_analysis_inputs(runs) is None


def test_step11_rejects_reused_native_log_path() -> None:
    """四个数值输入必须来自四个独立 native 日志，不能重复交接同一文件。"""

    valid = {
        "status": "completed",
        "ready": True,
        "software_status": "normal_termination",
        "convergence_status": "converged",
        "scientific_validation_status": "mechanically_valid",
        "log_path": "outputs/jobs/shared/stdout.log",
    }
    runs = {
        f"native_{label}_{stage}": dict(valid)
        for label in ("TS-2a", "TS-2b-1")
        for stage in ("step4_opt_freq", "step10_single_point")
    }

    assert collect_case1_fresh_analysis_inputs(runs) is None


def test_step11_ignores_irc_status_but_full_path_requires_it(tmp_path: Path) -> None:
    """IRC 不阻断数值后处理，但完整 Case 1 路径必须单独判为不完整。"""

    runs = _build_valid_native_runs(tmp_path)
    runs["native_TS-2a_step7_irc"]["status"] = "technical_pause"
    runs["native_TS-2a_step7_irc"]["minichem_status"] = "timeout"
    runs["native_TS-2a_step7_irc"]["software_status"] = "not_reached"
    runs["native_TS-2a_step7_irc"]["convergence_status"] = "not_reached"
    runs["native_TS-2a_step7_irc"]["scientific_validation_status"] = (
        "not_reached"
    )

    inputs = collect_case1_fresh_analysis_inputs(runs, operation_root=tmp_path)
    assert inputs is not None
    assert set(inputs) == {
        "ts_2a_step4_opt_freq",
        "ts_2a_step10_single_point",
        "ts_2b_1_step4_opt_freq",
        "ts_2b_1_step10_single_point",
    }
    full_path = evaluate_case1_full_native_path(runs, operation_root=tmp_path)
    assert full_path["status"] == "incomplete"
    assert full_path["scientifically_complete"] is False
    assert full_path["incomplete_stages"] == ["native_TS-2a_step7_irc"]


def test_full_native_path_accepts_chained_irc_and_single_point_records(
    tmp_path: Path,
) -> None:
    """完整路径需要同一 Step 4 checkpoint，并记录双向 IRC 证据。"""

    runs = _build_valid_native_runs(tmp_path)

    result = evaluate_case1_full_native_path(runs, operation_root=tmp_path)

    assert result["status"] == "completed"
    assert result["scientifically_complete"] is True
    assert result["incomplete_stages"] == []
    assert result["stage_results"]["native_TS-2a_step7_irc"]["dependency_link_status"] == (
        "validated"
    )
    assert result["stage_results"]["native_TS-2a_step7_irc"]["irc_connectivity_status"] == (
        "validated"
    )


def test_full_native_path_rejects_mismatched_downstream_checkpoint(tmp_path: Path) -> None:
    """下游引用另一条作业链时，完整路径不能被标为完成。"""

    runs = _build_valid_native_runs(tmp_path)
    runs["native_TS-2a_step7_irc"].update(
        {
            "parent_operation_id": "operation-wrong-parent",
            "parent_job_id": "job-wrong-parent",
            "parent_checkpoint_path": "outputs/wrong-parent.chk",
        }
    )

    result = evaluate_case1_full_native_path(runs, operation_root=tmp_path)

    assert result["status"] == "incomplete"
    assert result["scientifically_complete"] is False
    assert result["incomplete_stages"] == ["native_TS-2a_step7_irc"]
    assert result["stage_results"]["native_TS-2a_step7_irc"]["dependency_link_status"] == (
        "mismatch"
    )


def test_full_native_path_rejects_explicit_failed_irc_connectivity(tmp_path: Path) -> None:
    """IRC 明确报告端点未连通时，即使作业终态正常也不能通过。"""

    runs = _build_valid_native_runs(tmp_path)
    runs["native_TS-2a_step7_irc"]["irc_bidirectional_completed"] = False

    result = evaluate_case1_full_native_path(runs, operation_root=tmp_path)

    assert result["status"] == "incomplete"
    assert result["incomplete_stages"] == ["native_TS-2a_step7_irc"]
    assert result["stage_results"]["native_TS-2a_step7_irc"]["irc_connectivity_status"] == (
        "failed"
    )


def test_step11_rejects_invalid_step4_or_step10_log() -> None:
    """四份日志齐全也不能绕过数值依赖的机械科学有效性。"""

    valid = {
        "status": "completed",
        "ready": True,
        "software_status": "normal_termination",
        "convergence_status": "converged",
        "scientific_validation_status": "mechanically_valid",
    }
    runs = {}
    for label in ("TS-2a", "TS-2b-1"):
        for stage in ("step4_opt_freq", "step10_single_point"):
            runs[f"native_{label}_{stage}"] = {
                **valid,
                "outputs": [{"path": f"{label}/{stage}/stdout.log"}],
            }
    runs["native_TS-2a_step4_opt_freq"]["convergence_status"] = "failed"
    runs["native_TS-2a_step4_opt_freq"]["scientific_validation_status"] = (
        "mechanically_invalid"
    )

    assert collect_case1_fresh_analysis_inputs(runs) is None


def test_full_native_path_fails_closed_without_operation_and_dependency_identity() -> None:
    """缺少 operation/job/checkpoint、意图或 IRC 明确证据时不得宣称完整。"""

    valid = {
        "status": "completed",
        "ready": True,
        "software_status": "normal_termination",
        "convergence_status": "converged",
        "scientific_validation_status": "mechanically_valid",
        "outputs": [{"path": "shared/stdout.log"}],
    }
    runs = {
        f"native_{label}_{stage}": dict(valid)
        for label in ("TS-2a", "TS-2b-1")
        for stage in ("step4_opt_freq", "step7_irc", "step10_single_point")
    }

    result = evaluate_case1_full_native_path(runs)

    assert result["scientifically_complete"] is False
    assert result["status"] == "incomplete"
    assert result["incomplete_stages"]


def test_case1_completeness_summary_separates_capability_and_encapsulation() -> None:
    """manifest 必须分别报告工具能力、Action 封装粒度和回放边界。"""

    summary = build_case1_completeness_summary(
        {
            "service": {"status": "connected"},
            "discovery": {"candidate_actions": {}},
            "replay": {
                "full_path_validation": {
                    "status": "incomplete",
                    "scientifically_complete": False,
                },
                "deck_builder_coverage": {"2": {"separate_action": False}},
                "native_plan": [{"stage": "step4_opt_freq", "parent_operation_id": None}],
            },
        }
    )

    assert summary["tool_capability_completeness"]["status"] == "incomplete"
    assert summary["encapsulation_granularity_completeness"]["status"] == "partial"
    assert summary["encapsulation_granularity_completeness"]["shared_native_deck_builder_steps"] == [2]
    assert summary["replay_boundary"]["supplied_final_ts_coordinates"] is True
    assert summary["replay_boundary"]["raw_smiles_independent_discovery"] is False


def test_cli_require_complete_path_returns_nonzero_for_incomplete_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """显式要求完整路径时，服务连接但科学核验不完整也必须失败。"""

    monkeypatch.setattr(
        case1_tool_path,
        "run_case1_tool_path_validation",
        lambda *_args, **_kwargs: {
            "service": {"status": "connected"},
            "executions": {},
            "tool_capability_completeness": {"status": "incomplete"},
            "encapsulation_granularity_completeness": {"status": "partial"},
            "replay_boundary": {"raw_smiles_independent_discovery": False},
            "replay": {
                "full_path_validation": {
                    "status": "incomplete",
                    "scientifically_complete": False,
                }
            },
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_case1_tool_path.py", "--require-complete-path"],
    )

    assert case1_tool_path.main() == 3
