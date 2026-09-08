"""验证并发阶段 entry 的失败事实不会被后续成功事件覆盖。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from chemistry_multiagent.agents.hypothesis_agent import (
    HypothesisAgent,
    HypothesisProtocolError,
)
from chemistry_multiagent.utils.model_run_observer import ModelRunStageObserver
from chemistry_multiagent.utils.model_run_outputs import ModelRunOutputWriter
from chemistry_multiagent.utils.run_bundle import RunBundle
from chemistry_multiagent.utils.workflow_contract import (
    project_minichem_stage_tool_projection,
)
from tests.fakes.minichem_mcp import build_catalog_snapshot


def _read_json(path: Path) -> dict:
    """读取测试生成的 JSON 文档。"""

    return json.loads(path.read_text(encoding="utf-8"))


def test_later_validated_entry_does_not_overwrite_failed_stage() -> None:
    """同一阶段先失败、后成功完成的并发 entry 仍必须保留失败阶段状态。"""

    with TemporaryDirectory() as directory:
        bundle = RunBundle(
            Path(directory),
            research_task_id="failed-then-validated",
        )
        bundle.write_stage_summary(
            "hypothesis_preparation",
            {"status": "failed", "reason": "一个 synthesis 请求失败"},
            source_file="internal/audit/model_calls/merge-001.json",
            call_identity={"call_id": "merge-001"},
        )
        bundle.write_stage_summary(
            "hypothesis_preparation",
            {"status": "validated", "reason": "另一个 synthesis 请求成功"},
            source_file="internal/audit/model_calls/merge-002.json",
            call_identity={"call_id": "merge-002"},
        )

        stage = _read_json(bundle.root / "stages" / "hypothesis_preparation.json")
        summary = _read_json(bundle.summary_path)

        assert [entry["status"] for entry in stage["entries"]] == [
            "failed",
            "validated",
        ]
        assert stage["status"] == "failed"
        assert summary["stages"]["hypothesis_preparation"]["status"] == "failed"
        assert summary["stages"]["hypothesis_preparation"]["status_counts"] == {
            "failed": 1,
            "validated": 1,
        }


def _model_response(content: str) -> dict:
    """构造模型操作能够识别的最小成功响应。"""

    return {
        "id": "transport-failure-regression",
        "choices": [{"finish_reason": "stop", "message": {"content": content}}],
    }


class _SimulatedForbiddenError(Exception):
    """模拟不应在本测试中发起真实网络请求的 HTTP 403。"""

    status_code = 403


def _read_stage_input(messages: list[dict[str, str]]) -> dict:
    """读取统一模型消息中的阶段输入。"""

    return json.loads(messages[1]["content"])["base_input"]


def _partial_synthesis_failure_provider(messages: list[dict[str, str]]) -> dict:
    """生成两个独立 merge 组，并让含“假设1”的组固定返回 403。"""

    system_prompt = messages[0]["content"]
    stage_input = _read_stage_input(messages)
    if "研究策略师" in system_prompt:
        return _model_response(
            "one focused computational chemistry mechanism angle with a direct observable "
            "and a falsifiable trend across matched molecular models for rigorous comparison "
            "of geometry, electronic structure, energy profiles, ligand substitution effects, "
            "coordination stability, mechanistic alternatives, and reproducible low-cost "
            "quantum chemical validation criteria"
        )
    if "query_angle" in stage_input:
        return _model_response(
            json.dumps(
                [
                    {
                        "strategy_name": f"假设{i}",
                        "reasoning": f"假设{i}的可证伪计算判别。",
                    }
                    for i in range(1, 5)
                ],
                ensure_ascii=False,
            )
        )
    if "比较两项候选" in system_prompt:
        names = {
            stage_input["hypothesis_a"]["strategy_name"],
            stage_input["hypothesis_b"]["strategy_name"],
        }
        relationship = (
            "merge"
            if names in ({"假设1", "假设2"}, {"假设3", "假设4"})
            else "independent"
        )
        return _model_response(
            json.dumps(
                {
                    "relationship": relationship,
                    "reason": "最小探针只形成两个互不重叠的 merge 组。",
                },
                ensure_ascii=False,
            )
        )
    if "需要按指定关系合成" in system_prompt:
        names = {
            stage_input["hypothesis_a"]["strategy_name"],
            stage_input["hypothesis_b"]["strategy_name"],
        }
        if "假设1" in names:
            raise _SimulatedForbiddenError("simulated HTTP 403")
        # 让失败的 Synthesis 先结束，稳定复现“最后一个展示事件成功”的路径。
        time.sleep(0.05)
        return _model_response(
            json.dumps(
                {"strategy_name": "合成候选", "reasoning": "合成后的可证伪计算判别。"},
                ensure_ascii=False,
            )
        )
    raise AssertionError("测试 provider 收到未登记的模型阶段")


def test_partial_synthesis_transport_failure_marks_stage_failed_without_ranking() -> None:
    """部分 Synthesis 传输失败时，阶段不得完成且不得发布缺失的 Top5。"""

    with TemporaryDirectory() as directory:
        task_root = Path(directory)
        writer = ModelRunOutputWriter(task_root, "synthesis-transport-regression")
        observer = ModelRunStageObserver(writer, display_context={"batch_index": 1})
        agent = HypothesisAgent(
            comparison_workers=6,
            model_provider=_partial_synthesis_failure_provider,
        )

        try:
            agent.run_hypothesis_operation(
                task_root=str(task_root),
                operation_instance_id="hypothesis-operation-regression",
                scientific_question="检验配体取代如何改变镍催化反应选择性。",
                stage_tool_projection=project_minichem_stage_tool_projection(
                    build_catalog_snapshot()
                ),
                initial_query_angles=None,
                initial_retrieval_phase={"chemistry_context": {}},
                chemistry_context={},
                num_queries=1,
                num_hypotheses_per_query=4,
                top_n=2,
                hypothesis_ids_by_query=[["h1", "h2", "h3", "h4"]],
                merge_hypothesis_ids=["hm1", "hm2", "hm3"],
                observer=observer,
                display_context={"batch_index": 1},
            )
        except HypothesisProtocolError as error:
            assert error.failure_kind == "transport"
            operation_result = error.operation_result
            assert operation_result["status"] == "failed"
        else:  # pragma: no cover - 固定 provider 必须让本批失败
            raise AssertionError("Synthesis 传输失败没有传播到正式 operation 边界")

        stage_path = writer.output_root / "stages" / "hypothesis_preparation.json"
        stage_document = json.loads(stage_path.read_text(encoding="utf-8"))
        ranking_path = (
            writer.output_root
            / "03_hypothesis"
            / "batch_001"
            / "ranking"
            / "final_ranking.json"
        )

        assert [entry["status"] for entry in stage_document["entries"]][-2:] == [
            "failed",
            "validated",
        ]
        assert stage_document["status"] == "failed"
        assert not ranking_path.exists()
