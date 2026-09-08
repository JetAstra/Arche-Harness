"""Ranked Hypothesis 公开批次入口的结构和异常语义回归测试。"""

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


def _phase() -> dict[str, Any]:
    """构造一个最小但完整的规范 Hypothesis phase。"""

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
    """构造已验证 operation 及其正式清单引用。"""

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
    memory_enabled: bool = False,
    regeneration_package: dict[str, Any] | None = None,
    restart: dict[str, Any] | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """装配不启动外部模型或服务的批次上下文替身。"""

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
        load_hypothesis_restart=lambda: deepcopy(restart),
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
        task_config={
            "num_queries": 1,
            "num_hypotheses_per_query": 1,
            "top_n_hypotheses": 1,
        },
        identity_allocator=allocate,
        state=state,
        writer=None,
        case_profile=None,
        gaussian_route_policy=None,
        supplementary_scientific_input=None,
    )
    return context, saved, retrieval


def test_public_entry_keeps_initial_batch_stage_order_and_saved_facts(tmp_path: Path):
    """首批仍按 operation、排名、组合身份和原子保存顺序完成。"""

    context, saved, _retrieval = _context(tmp_path)
    calls: list[dict[str, Any]] = []
    agent = HypothesisAgent()

    def run_operation(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return _operation_result()

    agent.run_hypothesis_operation = run_operation  # type: ignore[method-assign]
    result = RankedHypothesisGenerator(agent).generate_ranked_hypothesis_batch(
        context, 1, None
    )

    assert result["status"] == "saved"
    assert len(calls) == 1
    assert calls[0]["initial_query_angles"] is None
    assert calls[0]["memory_enabled"] is None
    assert calls[0]["task_config_schema"] == "investigation_v2"
    assert saved["source_refs"] == [
        "internal/audit/hypothesis-operation.json"
    ]
    batch = saved["batch"]
    assert batch["generation_source"] == {
        "kind": "initial",
        "previous_batch_id": None,
        "source_record_ids": [],
    }
    assert batch["candidates"][0]["selection_status"] == "selected"
    assert batch["candidates"][0]["combination_id"] == "combination-1"


def test_public_entry_reuses_regeneration_memory_and_previous_batch(tmp_path: Path):
    """再生成批次继续携带冻结角度、Working Memory 和上一批身份。"""

    context, saved, retrieval = _context(
        tmp_path,
        batch_index=2,
        memory_enabled=True,
    )
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
        "generation_config": {
            "num_queries": 1,
            "num_hypotheses_per_query": 1,
            "top_n_hypotheses": 1,
        },
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
    RankedHypothesisGenerator(agent).generate_ranked_hypothesis_batch(
        context, 2, "batch-1"
    )

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


@pytest.mark.parametrize(
    ("batch_index", "previous_batch_id"),
    [(0, None), (1, "batch-0"), (2, None)],
)
def test_public_entry_rejects_invalid_batch_chain(
    tmp_path: Path,
    batch_index: int,
    previous_batch_id: str | None,
):
    """批次身份错误必须在读取 Retrieval 和调用模型前失败。"""

    context, _saved, _retrieval = _context(tmp_path)
    with pytest.raises(ValueError):
        RankedHypothesisGenerator(HypothesisAgent()).generate_ranked_hypothesis_batch(
            context, batch_index, previous_batch_id
        )


def test_protocol_failure_saves_pause_with_transport_reason(tmp_path: Path):
    """模型传输失败仍保存清单引用和可恢复暂停，而不伪造批次。"""

    context, saved, _retrieval = _context(tmp_path)
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


def test_validated_operation_without_manifest_still_fails_closed(tmp_path: Path):
    """已标记成功但缺少正式清单的 operation 不得写入批次事实。"""

    context, saved, _retrieval = _context(tmp_path)
    agent = HypothesisAgent()
    agent.run_hypothesis_operation = lambda **_kwargs: {  # type: ignore[method-assign]
        "operation_result": {"status": "validated"},
        "hypothesis_result": _phase(),
    }

    with pytest.raises(RuntimeError, match="正式清单"):
        RankedHypothesisGenerator(agent).generate_ranked_hypothesis_batch(
            context, 1, None
        )
    assert saved == {}
