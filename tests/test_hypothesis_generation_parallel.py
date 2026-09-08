"""Hypothesis 四 Query 并发生成的离线合同测试。"""

from __future__ import annotations

import copy
import json
import re
import threading
import time
from pathlib import Path

import pytest

from chemistry_multiagent.agents.hypothesis_agent import (
    HypothesisAgent,
    HypothesisProtocolError,
)
from chemistry_multiagent.controllers.case23_reproduction import (
    Case23ReproductionBudgets,
    _build_case23_cli_parser,
)
from chemistry_multiagent.controllers.chemistry_multiagent_controller import (
    _build_cli_parser as _build_controller_cli_parser,
)
from tests.test_hypothesis_batches import (
    _dynamic_input,
    _envelope,
    _long_query,
    _ticket03_chemical_capability_projection,
    _ticket03_stage_tool_projection,
)


def _parallel_arguments() -> dict:
    """构造固定四 Query、每 Query 三条正文的离线 operation 输入。"""

    return {
        "scientific_question": "解释目标分子的构象变化与振动光谱响应",
        "stage_tool_projection": _ticket03_stage_tool_projection(),
        "chemical_capability_projection": _ticket03_chemical_capability_projection(),
        "initial_query_angles": None,
        "initial_retrieval_phase": {
            "keywords": ["conformation spectroscopy"],
            "literature_review": "冻结文献摘要不能被子调用修改。",
            "mechanistic_clues": ["构象变化可能改变振动强度"],
            "limitations": ["缺少实验峰表"],
            "chemistry_context": {"charge": 0, "multiplicity": 1},
        },
        "chemistry_context": {"charge": 0, "multiplicity": 1},
        "num_queries": 4,
        "num_hypotheses_per_query": 3,
        "top_n": 5,
        "hypothesis_ids_by_query": [
            [f"h{query}{candidate}" for candidate in range(1, 4)]
            for query in range(1, 5)
        ],
        "merge_hypothesis_ids": [],
    }


def _query_index(dynamic: dict) -> int:
    """从测试 Query 文本提取编号，不读取运行身份。"""

    match = re.match(r"Q([1-4])\b", dynamic["query_angle"])
    assert match is not None
    return int(match.group(1))


def _query_response() -> dict:
    """返回四个固定研究角度，覆盖首批 Query stage。"""

    return _envelope("<>".join(_long_query(f"Q{index}") for index in range(1, 5)))


def _hypothesis_response(query_index: int) -> dict:
    """返回该 Query 的三条合法、互不重复的科学正文。"""

    return _envelope(
        json.dumps(
            [
                {
                    "strategy_name": f"Q{query_index}-strategy-{candidate}",
                    "reasoning": (
                        f"Q{query_index} 的第 {candidate} 条策略说明成立条件、"
                        "可观察差异以及可能的反证。"
                    ),
                }
                for candidate in range(1, 4)
            ],
            ensure_ascii=False,
        )
    )


def _new_parallel_agent(provider, *, workers: int = 4) -> HypothesisAgent:
    """创建只测试生成阶段的 Agent，屏蔽后续比较。"""

    agent = HypothesisAgent(
        comparison_workers=1,
        hypothesis_generation_workers=workers,
        model_provider=provider,
    )
    agent._optimize_hypotheses = (  # type: ignore[method-assign]
        lambda candidates, *_args, **_kwargs: candidates
    )
    agent._rank_strategies = (  # type: ignore[method-assign]
        lambda candidates, *_args, **_kwargs: candidates
    )
    return agent


def _run_parallel_operation(
    tmp_path: Path,
    agent: HypothesisAgent,
    operation_id: str,
):
    """运行正式动态 operation，覆盖预留、清单和 finalize。"""

    return agent.run_hypothesis_operation(
        task_root=str(tmp_path),
        operation_instance_id=operation_id,
        **_parallel_arguments(),
    )


def test_generation_worker_default_is_independent_from_comparison_worker() -> None:
    """通用生成默认串行，比较并发配置不会暗中改变它。"""

    default_agent = HypothesisAgent(comparison_workers=9)
    assert default_agent.hypothesis_generation_workers == 1
    configured_agent = HypothesisAgent(
        comparison_workers=1,
        hypothesis_generation_workers=2,
    )
    assert configured_agent.comparison_workers == 1
    assert configured_agent.hypothesis_generation_workers == 2


def test_case23_formal_budget_defaults_generation_to_four_and_allows_override() -> None:
    """正式四 Query 复现默认四路生成，并允许单独覆盖。"""

    assert Case23ReproductionBudgets().hypothesis_generation_workers == 4
    assert (
        Case23ReproductionBudgets(hypothesis_generation_workers=2)
        .validate_case23_reproduction_budget_values()
        .hypothesis_generation_workers
        == 2
    )
    with pytest.raises(ValueError):
        Case23ReproductionBudgets(hypothesis_generation_workers=0).validate_case23_reproduction_budget_values()
    args = _build_case23_cli_parser().parse_args(
        ["--case", "case2", "--hypothesis-generation-workers", "2"]
    )
    assert args.hypothesis_generation_workers == 2
    controller_args = _build_controller_cli_parser().parse_args(
        [
            "create-standard-research-task",
            "--question",
            "普通任务",
            "--hypothesis-generation-workers",
            "2",
        ]
    )
    assert controller_args.hypothesis_generation_workers == 2


def test_generation_collects_out_of_order_futures_as_q1_to_q4(tmp_path: Path) -> None:
    """完成顺序乱序时，Query 分组和预分配身份仍保持 Q1 到 Q4。"""

    completion_order: list[int] = []
    order_lock = threading.Lock()

    def provider(messages):
        dynamic = _dynamic_input(messages)
        if "query_angle" not in dynamic:
            return _query_response()
        query_index = _query_index(dynamic)
        time.sleep({1: 0.08, 2: 0.05, 3: 0.02, 4: 0.0}[query_index])
        with order_lock:
            completion_order.append(query_index)
        return _hypothesis_response(query_index)

    result = _run_parallel_operation(
        tmp_path,
        _new_parallel_agent(provider),
        "parallel-out-of-order",
    )

    assert completion_order != [1, 2, 3, 4]
    groups = result["hypothesis_result"]["hypotheses_by_query"]
    assert [group["query"] for group in groups] == [
        _long_query(f"Q{index}") for index in range(1, 5)
    ]
    assert [
        hypothesis["hypothesis_id"]
        for group in groups
        for hypothesis in group["hypotheses"]
    ] == [
        f"h{query}{candidate}"
        for query in range(1, 5)
        for candidate in range(1, 4)
    ]


def test_generation_respects_explicit_worker_cap(tmp_path: Path) -> None:
    """四个 Query 并发时，活动模型调用数不超过生成并发上限。"""

    active = 0
    maximum_active = 0
    state_lock = threading.Lock()

    def provider(messages):
        nonlocal active, maximum_active
        dynamic = _dynamic_input(messages)
        if "query_angle" not in dynamic:
            return _query_response()
        query_index = _query_index(dynamic)
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            time.sleep(0.04)
            return _hypothesis_response(query_index)
        finally:
            with state_lock:
                active -= 1

    result = _run_parallel_operation(
        tmp_path,
        _new_parallel_agent(provider, workers=2),
        "parallel-cap",
    )

    assert result["operation_result"]["status"] == "validated"
    assert maximum_active == 2


def test_parallel_generation_manifest_has_stable_reserved_stage_order(
    tmp_path: Path,
) -> None:
    """并行生成的清单按预留顺序记录 Query 及其尝试。"""

    def provider(messages):
        dynamic = _dynamic_input(messages)
        if "query_angle" not in dynamic:
            return _query_response()
        return _hypothesis_response(_query_index(dynamic))

    operation = _run_parallel_operation(
        tmp_path,
        _new_parallel_agent(provider),
        "parallel-manifest",
    )["operation_result"]

    assert operation["status"] == "validated"
    stages = operation["manifest"]["stages"]
    assert [stage["stage_index"] for stage in stages] == [1, 2, 3, 4, 5]
    assert [stage["stage"] for stage in stages] == [
        "query",
        "hypothesis",
        "hypothesis",
        "hypothesis",
        "hypothesis",
    ]
    assert [stage["query_index"] for stage in stages] == [None, 1, 2, 3, 4]
    assert all(stage["status"] == "validated" for stage in stages)
    assert all(len(stage["attempts"]) == 1 for stage in stages)
    assert Path(operation["manifest_path"]).is_file()


@pytest.mark.parametrize("failed_queries", [{3}, {2, 4}])
def test_parallel_generation_reports_minimum_failure_after_all_tasks_finish(
    tmp_path: Path,
    failed_queries: set[int],
) -> None:
    """单个或多个失败都等待全部预留 stage，并按最小 Query 失败。"""

    calls: list[int] = []

    def provider(messages):
        dynamic = _dynamic_input(messages)
        if "query_angle" not in dynamic:
            return _query_response()
        query_index = _query_index(dynamic)
        calls.append(query_index)
        if query_index in failed_queries:
            return _envelope("not-json")
        return _hypothesis_response(query_index)

    with pytest.raises(HypothesisProtocolError) as raised:
        _run_parallel_operation(
            tmp_path,
            _new_parallel_agent(provider),
            "parallel-failure",
        )

    error = raised.value
    assert error.query_index == min(failed_queries)
    stages = error.operation_result["manifest"]["stages"]
    assert [stage["query_index"] for stage in stages] == [None, 1, 2, 3, 4]
    assert [stage["status"] for stage in stages] == [
        "validated",
        *[
            "failed" if index in failed_queries else "validated"
            for index in range(1, 5)
        ],
    ]
    assert all(calls.count(index) >= 1 for index in range(1, 5))


def test_parallel_generation_recovery_reuses_only_validated_query_prefix(
    tmp_path: Path,
) -> None:
    """失败后恢复只重试首个失败 Query 及其后的 Query。"""

    def failing_provider(messages):
        dynamic = _dynamic_input(messages)
        if "query_angle" not in dynamic:
            return _query_response()
        if _query_index(dynamic) == 3:
            return _envelope("not-json")
        return _hypothesis_response(_query_index(dynamic))

    with pytest.raises(HypothesisProtocolError) as raised:
        _run_parallel_operation(
            tmp_path,
            _new_parallel_agent(failing_provider),
            "parallel-recovery-source",
        )
    prior_manifest_path = raised.value.operation_result["manifest_path"]

    retry_calls: list[int] = []

    def recovery_provider(messages):
        dynamic = _dynamic_input(messages)
        if "query_angle" not in dynamic:
            raise AssertionError("已验证的 Query stage 不应重新调用 provider")
        query_index = _query_index(dynamic)
        retry_calls.append(query_index)
        return _hypothesis_response(query_index)

    retry_result = _new_parallel_agent(recovery_provider).run_hypothesis_operation(
        task_root=str(tmp_path),
        operation_instance_id="parallel-recovery-retry",
        prior_manifest_path=prior_manifest_path,
        **_parallel_arguments(),
    )

    assert sorted(retry_calls) == [3, 4]
    assert retry_result["operation_result"]["status"] == "validated"
    assert retry_result["operation_result"]["manifest"]["prior_manifest_ref"]


def test_parallel_generation_keeps_each_query_input_isolated(tmp_path: Path) -> None:
    """一个 provider 修改本地消息对象时，不得污染其他 Query 输入。"""

    observed: dict[int, dict] = {}
    observed_lock = threading.Lock()

    def provider(messages):
        dynamic = _dynamic_input(messages)
        if "query_angle" not in dynamic:
            return _query_response()
        query_index = _query_index(dynamic)
        snapshot = copy.deepcopy(dynamic)
        dynamic["literature_review"] = f"被 Q{query_index} 修改"
        dynamic["chemical_capability_projection"][0]["description"] = (
            f"被 Q{query_index} 修改"
        )
        with observed_lock:
            observed[query_index] = snapshot
        return _hypothesis_response(query_index)

    _run_parallel_operation(
        tmp_path,
        _new_parallel_agent(provider),
        "parallel-input-isolation",
    )

    assert sorted(observed) == [1, 2, 3, 4]
    assert all(
        item["literature_review"] == "冻结文献摘要不能被子调用修改。"
        and item["chemical_capability_projection"][0]["description"]
        == _ticket03_chemical_capability_projection()[0]["description"]
        for item in observed.values()
    )


def test_generation_and_comparison_limits_remain_independent() -> None:
    """生成并发为 2 时，比较 helper 仍独立遵守 1 路上限。"""

    agent = HypothesisAgent(
        comparison_workers=1,
        hypothesis_generation_workers=2,
    )
    active = 0
    maximum_active = 0
    state_lock = threading.Lock()

    def comparison_worker(_payload):
        nonlocal active, maximum_active
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            time.sleep(0.02)
            return "ok"
        finally:
            with state_lock:
                active -= 1

    result = agent._run_indexed_comparisons_strict(
        [(index, index) for index in range(4)],
        comparison_worker,
    )
    assert result == [(index, "ok") for index in range(4)]
    assert maximum_active == 1
    assert agent.hypothesis_generation_workers == 2


class _StreamAfterBodyFailure(Exception):
    """模拟已收到流正文后才发生的传输失败。"""

    status_code = 403
    body = {"code": "insufficient_user_quota"}
    _stream_response_body_received = True


def test_streaming_after_body_failure_is_not_retried(tmp_path: Path) -> None:
    """流式正文后的失败不重发，但其他已提交 Query 仍完成并入清单。"""

    calls: list[int] = []

    def provider(messages):
        dynamic = _dynamic_input(messages)
        if "query_angle" not in dynamic:
            return _query_response()
        query_index = _query_index(dynamic)
        calls.append(query_index)
        if query_index == 2:
            raise _StreamAfterBodyFailure("stream ended after body")
        return _hypothesis_response(query_index)

    with pytest.raises(HypothesisProtocolError) as raised:
        _run_parallel_operation(
            tmp_path,
            _new_parallel_agent(provider),
            "parallel-stream-failure",
        )

    assert raised.value.query_index == 2
    assert calls.count(2) == 1
    assert sorted(calls) == [1, 2, 3, 4]
