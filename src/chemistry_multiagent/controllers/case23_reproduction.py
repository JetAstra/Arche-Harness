"""从冻结论文语料启动或恢复 Case 2/3 的正式端到端复现。

本模块只负责选择固定 Case、校验冻结语料并装配 Controller。Retrieval、Hypothesis、
Investigation and Experiment、Task Review 与 Publish 由
``ResearchTaskApplication.create_standard_research_task`` 的唯一公开入口
按当前 v2 顺序执行。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from chemistry_multiagent.agents.retrieval_agent import (
    RetrievalAgent,
    normalize_paper_search_config,
)
from chemistry_multiagent.controllers.chemistry_multiagent_controller import (
    ResearchTaskApplication,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_CASE23_FORMAL_PAPER_COUNT = 50


# 论文 SI C.6/C.7 的 Scientific Question Input 是 Case 2/3 的权威题面。
# 冻结论文 manifest 只描述已校验的语料，不应使用其中为检索而缩写的 question
# 覆盖这些题面。这里保留题面中的实验事实、背景和设计要求，但不加入作者后来
# 的 workflow、计算结果、坐标或最终机制。
CASE23_AUTHOR_SCIENTIFIC_QUESTION_INPUTS: Mapping[str, str] = MappingProxyType(
    {
        "case2": (
            "In my experiment, the compound Ph–CH2–CH2–CHI–BPin and HPPh2 were "
            "converted into Ph–CH2–CH2–CH2–BPin and IPPh2. The reaction was "
            "carried out with 2 equivalents of Cs2CO3 in a mixed solvent of "
            "EtOAc/CPME under 450 nm blue-light irradiation, affording the "
            "product in 83% yield.\n\n"
            "Question:\n"
            "Based on these observations, I would like to formulate mechanistic "
            "hypotheses (concerted, nucleophilic/electrophilic substitution, "
            "radical pathway, etc.). Please design an independent computational "
            "workflow to assess the feasibility of each proposed mechanism. For "
            "each hypothesis, please suggest a simple computational chemistry test "
            "to assess its thermodynamic or kinetic feasibility. Only the overall "
            "feasibility of the chemical transformation needs to be examined; a full "
            "intermediate-by-intermediate pathway is not required. If the reaction "
            "proceeds via a concerted mechanism, you need to illustrate how the "
            "reactants cooperate; simply locate the transition state, compute its "
            "energy, and compare it with the energies of the reactants and products.\n\n"
            "Note:\n"
            "- The reaction is performed under visible-light irradiation and may "
            "involve photo-induced or excited-state processes.\n"
            "- Both stepwise and concerted pathways should be considered, including "
            "mechanisms involving radical, ionic or closed-shell reactivity.\n"
            "- It should be borne in mind that the C-I bond can undergo homolysis "
            "under visible light.\n"
            "- HPPh2 is a competent H-atom donor.\n\n"
            "As part of the requested comparison of competing mechanisms under the "
            "stated Cs2CO3/HPPh2 conditions, please also investigate interactions "
            "within a possible CsPPh2 pathway and assess its feasibility using the "
            "same independent computational criteria, without assuming that it is "
            "the correct mechanism."
        ),
        "case3": (
            "In the nickel-catalyzed migratory cross-coupling reaction between alkyl "
            "electrophiles and aryl boronic acids, the addition of TBAB "
            "(tetrabutylammonium bromide) and a strong base LiOH in DMA solvent is "
            "required. When Ni(II) is used as the catalyst, the ligand structure has "
            "a significant influence on whether the reaction produces the migratory "
            "product or the in situ product, i.e., the ligand exerts a strong "
            "regulatory effect on reaction selectivity:\n"
            "– When the ortho position adjacent to the nitrogen (N) atom in "
            "2,2′-bipyridine is substituted with a methyl group (–CH3), the reaction "
            "almost exclusively yields the migratory product;\n"
            "– When unsubstituted 2,2′-bipyridine, or meta-/para-methyl substituted "
            "bipyridines without ortho substitution, are used as ligands, the "
            "reaction almost exclusively yields the in situ product.\n\n"
            "Background mechanistic knowledge (to be referenced in reasoning):\n"
            "– For Ni(II) catalysts, the stable coordination geometries are primarily "
            "square-planar (four-coordinate).\n"
            "– The nitrogen atoms in the 2,2′-bipyridine ligand possess lone pairs of "
            "electrons and can form stable chelation with the Ni center in a bidentate "
            "fashion, which is essential for maintaining catalyst activity.\n"
            "– In the reaction system, the alkyl electrophile is efficiently converted "
            "into alkyl bromide (R-Br) by the additive TBAB.\n"
            "– Under the combined action of the solvent DMA and the strong base LiOH, "
            "the alkyl bromide (R-Br) sequentially coordinates both the bromide (Br−) "
            "and a hydride (H−) from the adjacent (β) carbon onto the catalyst. The "
            "formation of this tetra-coordinated catalyst (bearing N-N, Br, and H "
            "ligands) is a prerequisite for the migration reaction.\n"
            "– Similar coupling reactions catalyzed by palladium (Pd) have been "
            "extensively studied using DFT calculations, which can be referenced for "
            "understanding ligand effects (e.g., steric hindrance and electronic "
            "effects on coordination geometry and reaction barriers).\n\n"
            "Task:\n"
            "– Based on the above observations and background, propose abstract "
            "mechanistic hypotheses explaining why ortho-substitution reverses product "
            "selectivity, and design minimal computational experiments to test them:\n"
            "– Without specifying the explicit substrates or computing full transition "
            "states, propose abstract mechanistic hypotheses that explain why ortho "
            "substitution on 2,2′-bipyridine reverses product selectivity.\n"
            "– Propose reasonable mechanistic hypotheses regarding the influence of "
            "ortho substitution on the 2,2′-bipyridine ligand on product selectivity, "
            "explicitly clarifying the core logic (e.g., how steric hindrance, "
            "electronic effects, or changes in coordination modes affect key reaction "
            "steps).\n"
            "– For each hypothesis, design simple computational chemistry validation "
            "experiments, specifying the computational targets, methods, and evaluation "
            "criteria.\n"
            "– The computational experiment design should focus on the core mechanistic "
            "steps where ligand substitution influences product selectivity (e.g., "
            "stability of ligand–Ni coordination), without deriving the full reaction "
            "pathway. It is sufficient to verify the thermodynamic or kinetic feasibility "
            "of the hypotheses.\n"
            "– Each hypothesis must not only define a measurable indicator but also "
            "consider how this indicator changes across a range of values.\n"
            "– The goal is to compare how ortho substitution vs. other substitution "
            "patterns alter the trend of this indicator as it varies, rather than only "
            "reporting a single static value.\n"
            "– The reasoning should explicitly link differences in the indicator’s trend "
            "or energy profile to the observed switch in product selectivity.\n\n"
            "Note:\n"
            "– Each hypothesis must identify a measurable indicator—geometric, electronic, "
            "or energetic—that can be computed solely from the tetra-coordinate Ni(II) "
            "catalyst (bearing N–N, Br, and H ligands).\n"
            "– The indicator must be:\n"
            "– Computable with low-cost quantum-chemical methods (e.g., coordinate scans, "
            "bond-length scans, bond-angle scans, analysis of energy or electronic "
            "structure trends).\n"
            "– Independent of explicit substrates or transition state searches\n"
            "– Interpretable, with a clear theoretical link between its value and the "
            "shift from migratory to in situ products\n"
            "– Testable, such that applying it across a small ligand set (unsubstituted "
            "bpy, ortho-substituted, meta-/para-substituted derivatives) yields trends "
            "consistent with the experimental selectivity.\n\n"
            "Design requirements:\n"
            "– For each hypothesis, specify:\n"
            "– The definition of the indicator and how to compute it\n"
            "– The predicted relationship between the indicator and product selectivity\n"
            "– A minimal computational validation experiment (systems to compute, level "
            "of theory, evaluation criteria)"
        ),
    }
)


@dataclass(frozen=True)
class Case23ReproductionSpec:
    """描述一个固定复现 Case 使用的内部路由和本地冻结语料位置。"""

    case_id: str
    case_profile: str
    manifest_relative_path: Path
    pdf_dir_relative_path: Path
    index_dir_relative_path: Path


@dataclass(frozen=True)
class FrozenCase23Corpus:
    """保存已经通过完整性校验、可交给 no-search Retrieval 的输入。"""

    spec: Case23ReproductionSpec
    question: str
    manifest_path: Path
    pdf_dir: Path
    index_dir: Path
    target_papers: int


@dataclass(frozen=True)
class Case23ReproductionBudgets:
    """保存一次正式端到端复现使用的模型候选、生成并发与修订规模。"""

    num_queries: int = 4
    num_hypotheses_per_query: int = 3
    top_n_hypotheses: int = 5
    max_parallel_investigations: int = 5
    comparison_workers: int | str = "auto"
    hypothesis_generation_workers: int = 4
    max_hypothesis_batches: int = 3
    max_investigation_repairs_per_combination: int = 3

    def validate_case23_reproduction_budget_values(
        self,
    ) -> "Case23ReproductionBudgets":
        """校验本对象的预算字段并返回自身，不写文件或启动外部调用。

        查询、假设、两类并发、批次和 repair 数量必须是正整数；repair 沿用现有
        1 到 3 的上限。协议固定选择五个假设，比较并发支持 0/auto 全并发。字段是
        布尔值、非整数或越界时抛出 ``ValueError``。
        """

        positive_fields = {
            "num_queries": self.num_queries,
            "num_hypotheses_per_query": self.num_hypotheses_per_query,
            "max_parallel_investigations": self.max_parallel_investigations,
            "hypothesis_generation_workers": self.hypothesis_generation_workers,
            "max_hypothesis_batches": self.max_hypothesis_batches,
            "max_investigation_repairs_per_combination": (
                self.max_investigation_repairs_per_combination
            ),
        }
        for field, value in positive_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} 必须是正整数")
        if self.max_parallel_investigations > 5:
            raise ValueError("max_parallel_investigations 必须位于 1 到 5 之间")
        if self.top_n_hypotheses != 5:
            raise ValueError("top_n_hypotheses 必须固定为 5")
        if self.max_investigation_repairs_per_combination > 3:
            raise ValueError(
                "max_investigation_repairs_per_combination 必须位于 1 到 3 之间"
            )
        comparison_workers = self.comparison_workers
        if isinstance(comparison_workers, bool):
            raise ValueError("comparison_workers 不能是布尔值")
        if isinstance(comparison_workers, str):
            if comparison_workers.casefold() != "auto":
                raise ValueError("comparison_workers 只能是正整数、0 或 auto")
        elif isinstance(comparison_workers, int):
            if comparison_workers < 0:
                raise ValueError("comparison_workers 不能是负数")
        else:
            raise ValueError("comparison_workers 必须是正整数、0 或 auto")
        return self


CASE23_REPRODUCTION_REGISTRY: Mapping[str, Case23ReproductionSpec] = (
    MappingProxyType(
        {
            "case2": Case23ReproductionSpec(
                case_id="case2",
                case_profile="case2_fixed_reproduction",
                manifest_relative_path=Path(
                    "local_data/papers/"
                    "case2-e2e-20260819-scholar-target50/manifest.json"
                ),
                pdf_dir_relative_path=Path(
                    "local_data/papers/case2-e2e-20260819-scholar-target50/papers"
                ),
                index_dir_relative_path=Path(
                    "local_data/papers/case2-e2e-20260819-scholar-target50/index"
                ),
            ),
            "case3": Case23ReproductionSpec(
                case_id="case3",
                case_profile="case3_fixed_reproduction",
                manifest_relative_path=Path(
                    "local_data/papers/"
                    "case3-e2e-20260819-scholar-target50/manifest.json"
                ),
                pdf_dir_relative_path=Path(
                    "local_data/papers/case3-e2e-20260819-scholar-target50/papers"
                ),
                index_dir_relative_path=Path(
                    "local_data/papers/case3-e2e-20260819-scholar-target50/index"
                ),
            ),
        }
    )
)


def _resolve_registered_case_path(repository_root: Path, relative_path: Path) -> Path:
    """把登记的仓库相对路径解析为绝对路径，并拒绝越出仓库的结果。"""

    root = repository_root.expanduser().resolve(strict=False)
    candidate = (root / relative_path).resolve(strict=False)
    if candidate != root and root not in candidate.parents:
        raise ValueError("Case 2/3 冻结语料路径越出仓库")
    return candidate


def load_frozen_case23_corpus(
    case_id: str,
    *,
    repository_root: Path | None = None,
) -> FrozenCase23Corpus:
    """读取指定 Case 的 manifest 并复用 Retrieval 的冻结语料校验。

    输入是 ``case2`` 或 ``case3``，以及测试时可替换的仓库根目录。函数读取
    manifest、检查完成状态、目标数量、PDF 内容与哈希，再确认目标数量等于
    实际数量且目录中没有额外 PDF。返回的问题固定取自论文 SI C.6/C.7 的
    ``CASE23_AUTHOR_SCIENTIFIC_QUESTION_INPUTS``；manifest 中可能为检索生成的
    缩写 ``question`` 只作为历史元数据，不会覆盖权威题面。文件缺失、manifest
    非法或语料不一致时抛出 ``ValueError``，不会创建任务。
    """

    if not isinstance(case_id, str) or case_id not in CASE23_REPRODUCTION_REGISTRY:
        raise ValueError("case_id 必须是 case2 或 case3")
    spec = CASE23_REPRODUCTION_REGISTRY[case_id]
    root = repository_root or _REPOSITORY_ROOT
    manifest_path = _resolve_registered_case_path(root, spec.manifest_relative_path)
    pdf_dir = _resolve_registered_case_path(root, spec.pdf_dir_relative_path)
    index_dir = _resolve_registered_case_path(root, spec.index_dir_relative_path)

    try:
        manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{case_id} 冻结论文 manifest 无法读取") from exc
    if not isinstance(manifest_value, Mapping):
        raise ValueError(f"{case_id} 冻结论文 manifest 必须是 JSON object")

    # 复用 Retrieval 正式 no-search 入口的 PDF 可读性、SHA-256、语料指纹与
    # accepted_papers 校验，避免调试入口另写一套较弱的冻结规则。
    validated_manifest = RetrievalAgent._validate_frozen_paper_manifest(
        str(manifest_path),
        str(pdf_dir),
        target_papers=_CASE23_FORMAL_PAPER_COUNT,
    )
    target = validated_manifest["target_papers"]
    actual = validated_manifest["actual_papers"]
    accepted_papers = validated_manifest.get("accepted_papers")
    corpus_files = validated_manifest.get("corpus_files")
    if (
        target != _CASE23_FORMAL_PAPER_COUNT
        or actual != _CASE23_FORMAL_PAPER_COUNT
        or not isinstance(accepted_papers, list)
        or len(accepted_papers) != _CASE23_FORMAL_PAPER_COUNT
        or len(
            {
                item.get("sha256")
                for item in accepted_papers
                if isinstance(item, Mapping) and isinstance(item.get("sha256"), str)
            }
        )
        != _CASE23_FORMAL_PAPER_COUNT
        or not isinstance(corpus_files, list)
        or len(corpus_files) != _CASE23_FORMAL_PAPER_COUNT
    ):
        raise ValueError(
            f"{case_id} 正式 no-search manifest 必须满足 target=accepted=corpus_files=磁盘 PDF=50"
        )

    disk_pdf_files = sorted(path for path in pdf_dir.glob("*.pdf") if path.is_file())
    if len(disk_pdf_files) != _CASE23_FORMAL_PAPER_COUNT:
        raise ValueError(
            f"{case_id} 正式 no-search manifest 必须包含恰好 50 个磁盘 PDF"
        )

    question = CASE23_AUTHOR_SCIENTIFIC_QUESTION_INPUTS[case_id]
    return FrozenCase23Corpus(
        spec=spec,
        question=question,
        manifest_path=manifest_path,
        pdf_dir=pdf_dir,
        index_dir=index_dir,
        target_papers=target,
    )


def start_case23_controller_reproduction(
    case_id: str,
    *,
    budgets: Case23ReproductionBudgets | None = None,
    expert_model_enabled: bool = True,
    repository_root: Path | None = None,
    controller_factory: Callable[..., Any] | None = None,
) -> Mapping[str, Any]:
    """以 Memory On 和 no-search 参数启动一次 Case 2/3 正式工作流。

    函数先完成冻结语料校验，再按显式专家开关创建对应 case profile 的
    Controller，最后只调用 ``create_standard_research_task``。运行会写入
    标准 Run Bundle，并可能调用模型与 MiniChem；任何前置校验失败时不会构造
    Controller。
    """

    frozen = load_frozen_case23_corpus(case_id, repository_root=repository_root)
    selected_budgets = (
        budgets or Case23ReproductionBudgets()
    ).validate_case23_reproduction_budget_values()
    root = (repository_root or _REPOSITORY_ROOT).expanduser().resolve(strict=False)
    factory = controller_factory or ResearchTaskApplication
    controller = factory(
        work_dir=str(
            root / "local_data" / "runtime" / "case23_deepseek" / case_id
        ),
        memory_enabled=True,
        hypothesis_comparison_workers=selected_budgets.comparison_workers,
        hypothesis_generation_workers=selected_budgets.hypothesis_generation_workers,
        expert_model_enabled=expert_model_enabled,
        case_profile=frozen.spec.case_profile,
    )
    paper_search = normalize_paper_search_config(
        {
            "enabled": False,
            "provider_strategy": "pypaperbot_scholar",
            "target_papers": frozen.target_papers,
            "manifest_path": str(frozen.manifest_path),
        },
        enabled=False,
    )
    return controller.create_standard_research_task(
        scientific_question=frozen.question,
        num_queries=selected_budgets.num_queries,
        num_hypotheses_per_query=selected_budgets.num_hypotheses_per_query,
        top_n_hypotheses=selected_budgets.top_n_hypotheses,
        pdf_dir=str(frozen.pdf_dir),
        index_dir=str(frozen.index_dir),
        search_papers=False,
        paper_search=paper_search,
        max_parallel_investigations=(
            selected_budgets.max_parallel_investigations
        ),
        max_investigation_repairs_per_combination=(
            selected_budgets.max_investigation_repairs_per_combination
        ),
        max_hypothesis_batches=selected_budgets.max_hypothesis_batches,
    )


def resume_case23_controller_reproduction(
    research_task_id: str,
    *,
    controller_factory: Callable[..., Any] | None = None,
) -> Mapping[str, Any]:
    """只按任务身份恢复同一 Run Bundle 中的 Case 2/3 工作流。

    本入口不接收 Case、问题、manifest、profile 或任务预算。Controller 会从
    TaskCreated、Working Memory 和原检查点恢复这些冻结事实；任务身份为空或
    恢复事实不完整时由本函数或 Controller 直接失败。
    """

    if not isinstance(research_task_id, str) or not research_task_id.strip():
        raise ValueError("research_task_id 必须是非空字符串")
    factory = controller_factory or ResearchTaskApplication
    controller = factory(memory_enabled=True)
    return controller.resume_research_task(research_task_id.strip())


def _build_case23_cli_parser() -> argparse.ArgumentParser:
    """声明 Case 2/3 调试入口允许的最小命令行参数。"""

    parser = argparse.ArgumentParser(
        description="从冻结论文库启动或恢复 Case 2/3 Controller 端到端复现"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--case",
        choices=tuple(CASE23_REPRODUCTION_REGISTRY),
        help="启动新的固定 Case 复现",
    )
    action.add_argument(
        "--resume-research-task",
        help="只按 research_task_id 恢复既有 Run Bundle",
    )
    parser.add_argument("--num-queries", type=int, default=None)
    parser.add_argument("--num-hypotheses-per-query", type=int, default=None)
    parser.add_argument("--top-n-hypotheses", type=int, default=None)
    parser.add_argument("--max-parallel-investigations", type=int, default=None)
    parser.add_argument(
        "--hypothesis-comparison-workers",
        type=_parse_case23_comparison_workers,
        default=None,
    )
    parser.add_argument(
        "--hypothesis-generation-workers",
        type=int,
        default=None,
        help="四个 Query 假设生成子调用的最大并发数（默认 4）",
    )
    parser.add_argument("--max-hypothesis-batches", type=int, default=None)
    parser.add_argument(
        "--max-investigation-repairs-per-combination",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--disable-expert-model",
        action="store_true",
        help="显式关闭 Gaussian 专家审查和完整结果评价",
    )
    return parser


def _parse_case23_comparison_workers(value: str) -> int | str:
    """解析 Case 2/3 假设比较并发，0/auto 表示任务数全并发。"""

    if isinstance(value, str) and value.casefold() == "auto":
        return "auto"
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "comparison_workers 必须是正整数、0 或 auto"
        ) from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("comparison_workers 不能是负数")
    return parsed


def run_case23_reproduction_from_args(
    argv: Sequence[str] | None = None,
    *,
    repository_root: Path | None = None,
    controller_factory: Callable[..., Any] | None = None,
) -> Mapping[str, Any]:
    """解析命令行并分派一次新任务或纯 research_task_id 恢复。"""

    parser = _build_case23_cli_parser()
    args = parser.parse_args(argv)
    budget_values = (
        args.num_queries,
        args.num_hypotheses_per_query,
        args.top_n_hypotheses,
        args.max_parallel_investigations,
        args.hypothesis_comparison_workers,
        args.hypothesis_generation_workers,
        args.max_hypothesis_batches,
        args.max_investigation_repairs_per_combination,
    )
    if args.resume_research_task:
        if any(value is not None for value in budget_values):
            parser.error(
                "--resume-research-task 不能重传问题、Case profile 或任务预算"
            )
        if args.disable_expert_model:
            parser.error("--resume-research-task 不能重传专家模型开关")
        return resume_case23_controller_reproduction(
            args.resume_research_task,
            controller_factory=controller_factory,
        )

    defaults = Case23ReproductionBudgets()
    budgets = Case23ReproductionBudgets(
        num_queries=(
            args.num_queries if args.num_queries is not None else defaults.num_queries
        ),
        num_hypotheses_per_query=(
            args.num_hypotheses_per_query
            if args.num_hypotheses_per_query is not None
            else defaults.num_hypotheses_per_query
        ),
        top_n_hypotheses=(
            args.top_n_hypotheses
            if args.top_n_hypotheses is not None
            else defaults.top_n_hypotheses
        ),
        max_parallel_investigations=(
            args.max_parallel_investigations
            if args.max_parallel_investigations is not None
            else defaults.max_parallel_investigations
        ),
        comparison_workers=(
            args.hypothesis_comparison_workers
            if args.hypothesis_comparison_workers is not None
            else defaults.comparison_workers
        ),
        hypothesis_generation_workers=(
            args.hypothesis_generation_workers
            if args.hypothesis_generation_workers is not None
            else defaults.hypothesis_generation_workers
        ),
        max_hypothesis_batches=(
            args.max_hypothesis_batches
            if args.max_hypothesis_batches is not None
            else defaults.max_hypothesis_batches
        ),
        max_investigation_repairs_per_combination=(
            args.max_investigation_repairs_per_combination
            if args.max_investigation_repairs_per_combination is not None
            else defaults.max_investigation_repairs_per_combination
        ),
    )
    return start_case23_controller_reproduction(
        args.case,
        budgets=budgets,
        expert_model_enabled=not args.disable_expert_model,
        repository_root=repository_root,
        controller_factory=controller_factory,
    )


def _project_case23_cli_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """提取研究员继续检查或恢复任务所需的命令行字段。"""

    visible_keys = (
        "status",
        "research_task_id",
        "memory_status",
        "output_files",
        "reason_code",
        "pause_id",
        "pause_ids",
        "restart_guide",
        "combination_results",
        "model_outputs",
        "paper_search_status",
    )
    projected = {key: result[key] for key in visible_keys if key in result}
    projected.setdefault("status", "unknown")
    projected.setdefault("research_task_id", None)
    return projected


def main(argv: Sequence[str] | None = None) -> int:
    """执行 Case 2/3 命令行请求并返回与正式 Controller 一致的退出状态。"""

    result = run_case23_reproduction_from_args(argv)
    print(json.dumps(_project_case23_cli_result(result), ensure_ascii=False))
    return 1 if result.get("status") in {"failed", "retrieval_incomplete"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
