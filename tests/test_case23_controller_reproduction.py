from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

# Case 2/3 入口会先加载 RetrievalAgent；必须让它先导入 SentenceTransformer，
# 再加载 PyMuPDF 的 C++ 扩展，避免标准环境中的 libstdc++ 解析顺序冲突。
from chemistry_multiagent.controllers import case23_reproduction as module
from chemistry_multiagent.utils.gaussian_routing import derive_gaussian_route_policy

import fitz


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_queries", 0),
        ("num_hypotheses_per_query", 0),
        ("max_parallel_investigations", 0),
        ("max_parallel_investigations", 6),
        ("max_hypothesis_batches", 0),
        ("top_n_hypotheses", 4),
        ("max_investigation_repairs_per_combination", 0),
        ("max_investigation_repairs_per_combination", -1),
        ("max_investigation_repairs_per_combination", 4),
        ("max_investigation_repairs_per_combination", True),
        ("comparison_workers", -1),
        ("comparison_workers", True),
        ("comparison_workers", "invalid"),
    ],
)
def test_case23_budget_contract_rejects_invalid_values(
    field: str,
    value: object,
) -> None:
    """Case 2/3 预算沿用 Investigation 固定 Top5、正 repair 和正并发合同。"""

    defaults = module.Case23ReproductionBudgets()
    values = {
        name: getattr(defaults, name)
        for name in defaults.__dataclass_fields__
    }
    values[field] = value
    with pytest.raises(ValueError):
        module.Case23ReproductionBudgets(**values).validate_case23_reproduction_budget_values()


@pytest.mark.parametrize("comparison_workers", [0, "auto", 2])
def test_case23_budget_contract_accepts_comparison_worker_modes(
    comparison_workers: int | str,
) -> None:
    """Case 2/3 比较并发支持全并发 auto/0 和正整数上限。"""

    budgets = module.Case23ReproductionBudgets(
        comparison_workers=comparison_workers,
    )
    assert budgets.validate_case23_reproduction_budget_values() is budgets


def _write_pdf(path: Path, text: str) -> str:
    """创建一个可被正式 Retrieval 校验器读取的单页 PDF，并返回哈希。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    document.new_page().insert_text((72, 72), text)
    document.save(path)
    document.close()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_frozen_case_fixture(
    repository_root: Path,
    case_id: str,
    *,
    question: str = "只来自冻结 manifest 的科学问题。",
    paper_count: int = 50,
) -> tuple[Path, dict[str, Any]]:
    """按 registry 的固定位置写入最小冻结语料，不依赖本机 local_data。"""

    spec = module.CASE23_REPRODUCTION_REGISTRY[case_id]
    pdf_dir = repository_root / spec.pdf_dir_relative_path
    corpus_files = []
    for index in range(paper_count):
        pdf_path = pdf_dir / f"paper-{index + 1}.pdf"
        sha256 = _write_pdf(pdf_path, f"case fixture {index + 1}")
        corpus_files.append(
            {
                "path": pdf_path.name,
                "sha256": sha256,
                "size": pdf_path.stat().st_size,
            }
        )
    corpus_files = sorted(corpus_files, key=lambda item: item["path"])
    corpus_fingerprint = hashlib.sha256(
        json.dumps(
            corpus_files,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": 2,
        "status": "completed",
        "question": question,
        "target_papers": paper_count,
        "actual_papers": paper_count,
        "target_reached": True,
        "accepted_papers": deepcopy(corpus_files),
        "corpus_files": corpus_files,
        "corpus_fingerprint": corpus_fingerprint,
    }
    manifest_path = repository_root / spec.manifest_relative_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest_path, manifest


def test_case23_registry_uses_only_fixed_profiles_and_requested_corpora() -> None:
    """Case registry 不能落入 Case 1 profile 或 supplementary 输入。"""

    assert set(module.CASE23_REPRODUCTION_REGISTRY) == {"case2", "case3"}
    assert module.CASE23_REPRODUCTION_REGISTRY["case2"].case_profile == (
        "case2_fixed_reproduction"
    )
    assert module.CASE23_REPRODUCTION_REGISTRY["case3"].case_profile == (
        "case3_fixed_reproduction"
    )
    assert str(
        module.CASE23_REPRODUCTION_REGISTRY["case2"].manifest_relative_path
    ).endswith("case2-e2e-20260819-scholar-target50/manifest.json")
    assert str(
        module.CASE23_REPRODUCTION_REGISTRY["case3"].manifest_relative_path
    ).endswith("case3-e2e-20260819-scholar-target50/manifest.json")
    assert all(
        "supplementary" not in str(spec.manifest_relative_path)
        for spec in module.CASE23_REPRODUCTION_REGISTRY.values()
    )


@pytest.mark.parametrize("case_id", ["case2", "case3"])
def test_author_question_keeps_catalog_gaussian_route_for_compatibility_profile(case_id: str) -> None:
    """Case2/3 兼容 profile 不改变冻结目录 Gaussian 路由记录。"""

    spec = module.CASE23_REPRODUCTION_REGISTRY[case_id]
    policy = derive_gaussian_route_policy(
        module.CASE23_AUTHOR_SCIENTIFIC_QUESTION_INPUTS[case_id],
        case_profile=spec.case_profile,
    )

    assert policy.policy_id == "catalog_declared"
    assert policy.case_profile == spec.case_profile


@pytest.mark.parametrize("case_id", ["case2", "case3"])
def test_frozen_case_loader_uses_author_question_and_validates_pdf(
    tmp_path: Path,
    case_id: str,
) -> None:
    """完整论文题面覆盖 manifest 缩写，同时继续校验冻结论文。"""

    manifest_path, _manifest = _write_frozen_case_fixture(tmp_path, case_id)

    frozen = module.load_frozen_case23_corpus(
        case_id,
        repository_root=tmp_path,
    )

    assert frozen.question == module.CASE23_AUTHOR_SCIENTIFIC_QUESTION_INPUTS[case_id]
    assert frozen.question != "只来自冻结 manifest 的科学问题。"
    assert frozen.manifest_path == manifest_path.resolve()
    assert frozen.target_papers == 50
    assert len(list(frozen.pdf_dir.glob("*.pdf"))) == 50


@pytest.mark.parametrize(
    ("field", "value", "error_pattern"),
    [
        ("status", "partial", "尚未冻结"),
        ("target_reached", False, "尚未冻结"),
    ],
)
def test_frozen_case_loader_fails_closed_before_controller(
    tmp_path: Path,
    field: str,
    value: Any,
    error_pattern: str,
) -> None:
    """冻结标记缺失时立即失败，不进入任何在线阶段。"""

    manifest_path, manifest = _write_frozen_case_fixture(tmp_path, "case2")
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=error_pattern):
        module.load_frozen_case23_corpus("case2", repository_root=tmp_path)


@pytest.mark.parametrize("case_id", ["case2", "case3"])
def test_frozen_case_loader_does_not_require_manifest_question(
    tmp_path: Path,
    case_id: str,
) -> None:
    """论文题面是权威输入，冻结 manifest 不必携带检索用 question。"""

    manifest_path, manifest = _write_frozen_case_fixture(tmp_path, case_id)
    manifest.pop("question")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    frozen = module.load_frozen_case23_corpus(case_id, repository_root=tmp_path)

    assert frozen.question == module.CASE23_AUTHOR_SCIENTIFIC_QUESTION_INPUTS[case_id]


def test_case2_author_question_contains_complete_c6_and_natural_follow_up() -> None:
    """Case 2 保留 C.6 全部题面，并顺序融合额外用户回合的语义。"""

    question = module.CASE23_AUTHOR_SCIENTIFIC_QUESTION_INPUTS["case2"]

    for fragment in (
        "Ph–CH2–CH2–CHI–BPin",
        "HPPh2",
        "Ph–CH2–CH2–CH2–BPin",
        "IPPh2",
        "2 equivalents of Cs2CO3",
        "EtOAc/CPME",
        "450 nm blue-light irradiation",
        "83% yield",
        "concerted, nucleophilic/electrophilic substitution, radical pathway",
        "Only the overall feasibility of the chemical transformation",
        "photo-induced or excited-state processes",
        "C-I bond can undergo homolysis",
        "HPPh2 is a competent H-atom donor",
        "possible CsPPh2 pathway",
        "without assuming that it is the correct mechanism",
    ):
        assert fragment in question
    assert question.index("HPPh2 is a competent H-atom donor") < question.index(
        "possible CsPPh2 pathway"
    )
    for forbidden in ("dark", "TEMPO", "I2", "64.0 kcal", "EDA"):
        assert forbidden.lower() not in question.lower()


def test_case3_author_question_contains_complete_c7_requirements() -> None:
    """Case 3 题面包含 SI C.7 的背景、任务、判据和设计要求。"""

    question = module.CASE23_AUTHOR_SCIENTIFIC_QUESTION_INPUTS["case3"]

    for fragment in (
        "TBAB (tetrabutylammonium bromide)",
        "strong base LiOH in DMA solvent",
        "2,2′-bipyridine",
        "almost exclusively yields the migratory product",
        "almost exclusively yields the in situ product",
        "square-planar (four-coordinate)",
        "stable chelation with the Ni center in a bidentate fashion",
        "alkyl bromide (R-Br)",
        "bromide (Br−)",
        "hydride (H−) from the adjacent (β) carbon",
        "tetra-coordinated catalyst (bearing N-N, Br, and H ligands)",
        "Without specifying the explicit substrates or computing full transition states",
        "steric hindrance, electronic effects, or changes in coordination modes",
        "measurable indicator",
        "across a range of values",
        "small ligand set",
        "Design requirements:",
        "The definition of the indicator and how to compute it",
        "The predicted relationship between the indicator and product selectivity",
        "A minimal computational validation experiment",
    ):
        assert fragment in question
    assert "dihedral descriptor" not in question
    assert "ortho = lower" not in question.lower()


def test_frozen_case_loader_requires_target_equal_actual(tmp_path: Path) -> None:
    """正式 Case2/3 固定复现拒绝不是 50 篇的冻结 target。"""

    manifest_path, manifest = _write_frozen_case_fixture(
        tmp_path,
        "case2",
        paper_count=50,
    )
    manifest["target_papers"] = 49
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="目标数量与 paper_search 配置不一致|正式 no-search"):
        module.load_frozen_case23_corpus("case2", repository_root=tmp_path)


def test_frozen_case_loader_rejects_extra_unvalidated_pdf(tmp_path: Path) -> None:
    """目录中额外的无效 PDF 不能被正式去重收集器静默跳过。"""

    _write_frozen_case_fixture(tmp_path, "case2")
    spec = module.CASE23_REPRODUCTION_REGISTRY["case2"]
    (tmp_path / spec.pdf_dir_relative_path / "unexpected.pdf").write_bytes(b"invalid")

    with pytest.raises(ValueError, match="正式 no-search manifest 必须包含恰好 50 个磁盘 PDF"):
        module.load_frozen_case23_corpus("case2", repository_root=tmp_path)


def test_new_case_calls_only_canonical_controller_entry_with_formal_defaults(
    tmp_path: Path,
) -> None:
    """新任务固定 Memory On、no-search、4×3 假设预算和对应 profile。"""

    _write_frozen_case_fixture(tmp_path, "case2")
    calls: list[tuple[str, Any]] = []

    class _OfflineController:
        """替代完整在线边界并记录 Controller 装配与调用参数。"""

        def __init__(self, **kwargs: Any) -> None:
            calls.append(("init", deepcopy(kwargs)))

        def create_standard_research_task(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(("create_standard_research_task", deepcopy(kwargs)))
            return {"status": "open", "research_task_id": "research-task-case2"}

    result = module.start_case23_controller_reproduction(
        "case2",
        repository_root=tmp_path,
        controller_factory=_OfflineController,
    )

    assert result == {"status": "open", "research_task_id": "research-task-case2"}
    init = calls[0][1]
    assert init["memory_enabled"] is True
    assert init["hypothesis_comparison_workers"] == "auto"
    assert init["hypothesis_generation_workers"] == 4
    assert init["case_profile"] == "case2_fixed_reproduction"
    assert init["work_dir"] == str(
        tmp_path / "local_data/runtime/case23_deepseek/case2"
    )
    assert "memory_db_path" not in init
    request = calls[1][1]
    assert calls[1][0] == "create_standard_research_task"
    assert request["scientific_question"] == (
        module.CASE23_AUTHOR_SCIENTIFIC_QUESTION_INPUTS["case2"]
    )
    assert request["search_papers"] is False
    assert request["paper_search"]["enabled"] is False
    assert request["paper_search"]["target_papers"] == 50
    assert request["num_queries"] == 4
    assert request["num_hypotheses_per_query"] == 3
    assert request["top_n_hypotheses"] == 5
    assert request["max_parallel_investigations"] == 5
    assert request["max_hypothesis_batches"] == 3
    assert request["max_investigation_repairs_per_combination"] == 3
    assert "main_force_field" not in json.dumps(request)
    assert "target_force_field" not in json.dumps(request)


def test_new_case_cli_can_disable_expert_model(tmp_path: Path) -> None:
    """Case 2/3 固定入口显式关闭专家模型，并把选择传给 Controller。"""

    _write_frozen_case_fixture(tmp_path, "case3")
    calls: list[tuple[str, Any]] = []

    class _OfflineController:
        """记录新任务的 Controller 配置，不访问模型或 MiniChem。"""

        def __init__(self, **kwargs: Any) -> None:
            calls.append(("init", deepcopy(kwargs)))

        def create_standard_research_task(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(("create_standard_research_task", deepcopy(kwargs)))
            return {"status": "open", "research_task_id": "research-task-case3"}

    result = module.run_case23_reproduction_from_args(
        ["--case", "case3", "--disable-expert-model"],
        repository_root=tmp_path,
        controller_factory=_OfflineController,
    )

    assert result == {"status": "open", "research_task_id": "research-task-case3"}
    assert calls[0][0] == "init"
    assert calls[0][1]["expert_model_enabled"] is False


def test_resume_calls_application_only_with_research_task_id_and_memory_enabled() -> None:
    """恢复不重传问题、manifest、profile、预算，也不重新选择 Case。"""

    calls: list[tuple[str, Any]] = []

    class _OfflineController:
        """记录恢复装配参数，避免测试访问在线模型、工具或本地语料。"""

        def __init__(self, **kwargs: Any) -> None:
            calls.append(("init", deepcopy(kwargs)))

        def resume_research_task(self, research_task_id: str) -> dict[str, Any]:
            calls.append(("resume_research_task", research_task_id))
            return {"status": "paused", "research_task_id": research_task_id}

    result = module.resume_case23_controller_reproduction(
        "research-task-case23",
        controller_factory=_OfflineController,
    )

    assert result == {
        "status": "paused",
        "research_task_id": "research-task-case23",
    }
    assert calls == [
        ("init", {"memory_enabled": True}),
        ("resume_research_task", "research-task-case23"),
    ]


def test_case23_cli_resume_dispatches_only_research_task_id() -> None:
    """Case 2/3 CLI 恢复分支只创建 Memory On Application 并传 research_task_id。"""

    calls: list[tuple[str, Any]] = []

    class _OfflineController:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(("init", kwargs))

        def resume_research_task(self, research_task_id: str) -> dict[str, Any]:
            calls.append(("resume", research_task_id))
            return {"status": "succeeded", "research_task_id": research_task_id}

    result = module.run_case23_reproduction_from_args(
        ["--resume-research-task", "research-task-cli"],
        controller_factory=_OfflineController,
    )

    assert result == {
        "status": "succeeded",
        "research_task_id": "research-task-cli",
    }
    assert calls == [
        ("init", {"memory_enabled": True}),
        ("resume", "research-task-cli"),
    ]


def test_cli_forbids_question_override_and_resume_budget_replacement() -> None:
    """命令行不公开问题覆盖，恢复时也不能换用新的正式预算。"""

    with pytest.raises(SystemExit):
        module.run_case23_reproduction_from_args(
            ["--case", "case2", "--question", "覆盖问题"]
        )
    with pytest.raises(SystemExit):
        module.run_case23_reproduction_from_args(
            ["--resume-research-task", "research-task-case23", "--num-queries", "2"]
        )

@pytest.mark.parametrize("worker_value", ["auto", "0", "2"])
def test_case23_cli_accepts_comparison_worker_modes_without_corpus_access(
    worker_value: str,
) -> None:
    """Case 2/3 CLI 先解析 auto/0/正整数并把非法 corpus 留给后续阶段。"""

    with pytest.raises(ValueError, match="冻结论文"):
        module.run_case23_reproduction_from_args(
            [
                "--case",
                "case2",
                "--hypothesis-comparison-workers",
                worker_value,
            ],
            repository_root=Path("/tmp/nonexistent-case23-repository"),
        )


def test_case23_cli_rejects_negative_comparison_workers() -> None:
    """Case 2/3 CLI 在加载冻结语料前拒绝负比较并发。"""

    with pytest.raises(SystemExit):
        module.run_case23_reproduction_from_args(
            [
                "--case",
                "case2",
                "--hypothesis-comparison-workers",
                "-1",
            ],
            repository_root=Path("/tmp/nonexistent-case23-repository"),
        )
