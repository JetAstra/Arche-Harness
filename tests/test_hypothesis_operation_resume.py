"""Investigation V2 假设批次恢复和再生成边界回归。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chemistry_multiagent.agents.hypothesis_agent import (
    HypothesisAgent,
    HypothesisProtocolError,
    RankedHypothesisGenerator,
)
from chemistry_multiagent.memory.projections import (
    project_hypothesis_regeneration_context,
)
from tests.fakes.minichem_mcp import build_catalog_snapshot


TASK_CONFIG = {
    "num_queries": 1,
    "num_hypotheses_per_query": 1,
    "top_n_hypotheses": 1,
}


def _phase() -> dict[str, Any]:
    """构造一个已通过 V2 operation 校验的假设阶段。"""

    hypothesis = {
        "hypothesis_id": "hypothesis-model-id",
        "strategy_name": "受控计算策略",
        "reasoning": "使用冻结工具和条件验证当前科学问题。",
    }
    return {
        "queries": ["冻结研究角度"],
        "hypotheses_by_query": [
            {"query": "冻结研究角度", "hypotheses": [hypothesis]}
        ],
        "ranked_strategies": [
            {**hypothesis, "rank": 1, "score": 1, "wins": 1, "losses": 0}
        ],
    }


def _operation_result() -> dict[str, Any]:
    """构造带正式清单引用的 operation 返回值。"""

    return {
        "operation_result": {
            "status": "validated",
            "manifest_ref": {
                "relative_path": "internal/audit/hypothesis-operation.json"
            },
        },
        "hypothesis_result": _phase(),
    }


def _context(
    tmp_path: Path,
    *,
    batch_index: int = 1,
    memory_enabled: bool | None = None,
    regeneration_package: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """装配不启动外部模型的当前 V2 批次上下文。"""

    retrieval = {
        "keywords": ["冻结检索"],
        "literature_review": "冻结的检索结论",
        "mechanistic_clues": ["冻结线索"],
        "limitations": [],
        "chemistry_context": {"solvent": "water"},
    }
    saved: dict[str, Any] = {}
    counters: dict[str, int] = {}

    def allocate(prefix: str) -> str:
        counters[prefix] = counters.get(prefix, 0) + 1
        return f"{prefix}-{counters[prefix]}"

    def save_batch(payload: dict[str, Any], source_refs: list[str]) -> dict[str, Any]:
        saved["batch"] = deepcopy(payload)
        saved["source_refs"] = list(source_refs)
        return {"status": "saved", "batch": deepcopy(payload)}

    def save_pause(payload: dict[str, Any], source_refs: list[str]) -> dict[str, Any]:
        saved["pause"] = deepcopy(payload)
        saved["pause_source_refs"] = list(source_refs)
        return {"status": "paused", **payload, "source_refs": list(source_refs)}

    state = SimpleNamespace(
        memory_enabled=memory_enabled,
        load_retrieval_context=lambda: deepcopy(retrieval),
        load_hypothesis_restart=lambda: None,
        build_hypothesis_regeneration_package=lambda _index: deepcopy(
            regeneration_package
        ),
        save_ranked_hypothesis_batch=save_batch,
        save_hypothesis_pause=save_pause,
    )
    context = SimpleNamespace(
        task_id="task-1",
        task_root=tmp_path,
        scientific_question="验证冻结假设。",
        scientific_context={"chemistry_context": {"solvent": "water"}},
        catalog_snapshot=build_catalog_snapshot(),
        task_config=TASK_CONFIG,
        identity_allocator=allocate,
        state=state,
        writer=None,
        case_profile=None,
        gaussian_route_policy=None,
        supplementary_scientific_input=None,
    )
    return context, saved


def test_hypothesis_pause_saves_current_operation_boundary(tmp_path: Path) -> None:
    """模型协议失败只保存清单引用和结构化暂停，不伪造空批次。"""

    context, saved = _context(tmp_path)
    agent = HypothesisAgent()

    def fail_operation(**_kwargs: Any) -> Any:
        error = HypothesisProtocolError("hypothesis", "transport")
        error.manifest_ref = {
            "relative_path": "internal/audit/failed-operation.json"
        }
        raise error

    agent.run_hypothesis_operation = fail_operation  # type: ignore[method-assign]
    result = RankedHypothesisGenerator(agent).generate_ranked_hypothesis_batch(
        context, 1, None
    )

    assert result["status"] == "paused"
    assert saved["pause"]["reason_code"] == "model_service_failure"
    assert saved["pause_source_refs"] == [
        "internal/audit/failed-operation.json"
    ]
    assert "batch" not in saved


def test_regeneration_reuses_only_frozen_context_and_source_refs(
    tmp_path: Path,
) -> None:
    """再生成继续使用冻结科学上下文，并把当前批次来源明确标为 Working Memory。"""

    context, saved = _context(tmp_path, batch_index=2, memory_enabled=True)
    retrieval = context.state.load_retrieval_context()
    scientific_context = project_hypothesis_regeneration_context(
        scientific_question=context.scientific_question,
        initial_query_angles=["冻结研究角度"],
        initial_retrieval_phase=retrieval,
        chemistry_context={"solvent": "water"},
    )
    package = {
        "scientific_context": scientific_context,
        "current_query": "冻结研究角度",
        "prior_hypothesis_outcomes": [],
        "next_batch_index": 2,
        "generation_config": TASK_CONFIG,
        "source_record_ids": ["wm-record-1"],
    }
    context.state.build_hypothesis_regeneration_package = lambda _index: deepcopy(
        package
    )
    calls: list[dict[str, Any]] = []
    agent = HypothesisAgent()

    def run_operation(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return _operation_result()

    agent.run_hypothesis_operation = run_operation  # type: ignore[method-assign]
    result = RankedHypothesisGenerator(agent).generate_ranked_hypothesis_batch(
        context, 2, "batch-1"
    )

    assert result["status"] == "saved"
    assert calls[0]["initial_query_angles"] == ["冻结研究角度"]
    assert calls[0]["memory_enabled"] is True
    assert calls[0]["regeneration_package"]["source_record_ids"] == [
        "wm-record-1"
    ]
    assert saved["batch"]["generation_source"] == {
        "kind": "working_memory_regeneration",
        "previous_batch_id": "batch-1",
        "source_record_ids": ["wm-record-1"],
    }


def test_regeneration_package_is_deep_copied_and_batch_chain_is_strict(
    tmp_path: Path,
) -> None:
    """调用方修改再生成包不能污染冻结输入，非法批次链在模型调用前失败。"""

    context, _saved = _context(tmp_path, batch_index=2, memory_enabled=True)
    retrieval = context.state.load_retrieval_context()
    package = {
        "scientific_context": project_hypothesis_regeneration_context(
            scientific_question=context.scientific_question,
            initial_query_angles=["冻结研究角度"],
            initial_retrieval_phase=retrieval,
            chemistry_context={"solvent": "water"},
        ),
        "current_query": "冻结研究角度",
        "prior_hypothesis_outcomes": [],
        "next_batch_index": 2,
        "generation_config": TASK_CONFIG,
        "source_record_ids": [],
    }
    context.state.build_hypothesis_regeneration_package = lambda _index: deepcopy(
        package
    )
    agent = HypothesisAgent()
    captured: list[dict[str, Any]] = []
    agent.run_hypothesis_operation = lambda **kwargs: (
        captured.append(kwargs) or _operation_result()
    )  # type: ignore[method-assign]

    result = RankedHypothesisGenerator(agent).generate_ranked_hypothesis_batch(
        context, 2, "batch-1"
    )
    assert result["status"] == "saved"
    captured[0]["regeneration_package"]["scientific_context"]["chemistry_context"][
        "solvent"
    ] = "changed"
    assert (
        package["scientific_context"]["chemistry_context"]["solvent"] == "water"
    )

    for batch_index, previous_batch_id in ((0, None), (1, "batch-0"), (2, None)):
        bad_context, _ = _context(tmp_path / f"bad-{batch_index}", batch_index=batch_index)
        with pytest.raises(ValueError):
            RankedHypothesisGenerator(HypothesisAgent()).generate_ranked_hypothesis_batch(
                bad_context, batch_index, previous_batch_id
            )
