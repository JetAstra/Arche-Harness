"""Investigation and Experiment 阶段的确定性请求构造器。

本模块为 Investigation 的首次、返修、异步作业 continuation 和任务级 Review
边界生成可直接交给 Harness 的请求映射。动态
事实一律先经过严格 JSON 和路径检查，再以排序后的 JSON 放入 user message；构造
过程不读取文件、不访问网络，也不会创建或恢复会话。
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

from .contracts import (
    _assert_no_path_or_secret,
    _canonicalize_json_value,
    build_investigation_provider_json_schema,
    build_task_finalization_provider_json_schema,
    build_task_review_provider_json_schema,
    require_safe_public_scientific_context,
    require_safe_public_scientific_text,
    validate_investigation_response,
)
from chemistry_multiagent.utils.scientific_input import (
    normalize_initial_calculation_structures,
    normalize_initial_transition_state_structures,
)


__all__ = [
    "INVESTIGATION_COMPACT_PROMPT",
    "INVESTIGATION_COMPACT_PROMPT_SHA256",
    "INVESTIGATION_MODEL_CONTEXT_WINDOW",
    "build_initial_investigation_request",
    "build_repair_investigation_request",
    "build_investigation_job_wake_request",
    "build_investigation_job_continuation_request",
    "build_task_review_request",
    "build_task_finalization_request",
]


_DEFAULT_INVESTIGATION_MODEL = "gpt-5.6-sol"
_DEFAULT_REVIEW_MODEL = "deepseek-v4-pro"
INVESTIGATION_MODEL_CONTEXT_WINDOW = 1_000_000
INVESTIGATION_COMPACT_PROMPT = """
You are compacting the history of an ongoing chemistry investigation. Produce a faithful, self-contained state snapshot that allows the investigation model to continue without access to the earlier conversation. You are a recorder, not an investigator.

Use concise Markdown in the primary language of the investigation, with exactly two top-level sections:

## Scientific reasoning

Preserve the scientific question, researcher-provided hypotheses and constraints, the reasoning developed so far, the evidence supporting each line of reasoning, unresolved alternatives and contradictions, uncertainties, and decisions already made. Preserve rejected or superseded lines briefly together with the original reason they were rejected or superseded. Preserve the already-decided continuation point, if one exists.

## Computational jobs and results

For every submitted computation, preserve what was computed and why, all non-secret identifiers needed to query or retrieve that same computation, its latest trustworthy state, and an objective summary of any returned result. Preserve genuine computation failures, non-convergence, missing results, and result-collection failures.

Collapse repeated polling of the same job into its latest trustworthy state. Remove intermediate polls, repeated waiting messages, and recovered transient query errors. If reported states conflict, preserve the conflict instead of choosing one.

Rules:

- Report only reasoning, decisions, results, and uncertainties already present in the history.
- Do not introduce new scientific inferences, rank hypotheses, change confidence, resolve contradictions, discard unresolved alternatives, declare completion, or choose the next computation.
- Clearly preserve the distinction between tool-returned facts, model interpretations, and unverified hypotheses.
- Preserve identifiers, evidence references, chemical objects, numerical values, signs, units, and conditions exactly when they appear in the summary. Do not silently correct, round, or invent them.
- You may merge duplicate statements and replace large retrievable tool payloads with faithful objective summaries. Do not delete distinct scientific reasoning, results, contradictions, failures, or unresolved questions merely to make the summary shorter.
- Produce a complete current snapshot on every compaction. Never refer to an earlier summary or assume that any earlier conversation will remain available.
- Do not reproduce authentication credentials, secrets, unrelated infrastructure details, base instructions, tool schemas, repeated logs, or irrelevant internal paths. Ordinary job and result-retrieval identifiers are not authentication secrets and must be preserved.
- Be concise, but use as much space as necessary to preserve the investigation faithfully.
""".strip()
INVESTIGATION_COMPACT_PROMPT_SHA256 = hashlib.sha256(
    INVESTIGATION_COMPACT_PROMPT.encode("utf-8")
).hexdigest()
_HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MATERIAL_PATH_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_MATERIAL_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_CASE3_ADVISORY_CONSTRAINT_SCHEMA = (
    "arche.case3.author_main_hypothesis.constraint_semantics.v1"
)
_INVESTIGATION_JOB_TOOL_FAMILIES = {
    "transition_state_path": (
        "submit_action_job",
        "get_execution_job",
        "collect_execution_job",
    ),
    "native_execution": (
        "submit_native_job",
        "get_execution_job",
        "collect_execution_job",
    ),
    "action_execution": (
        "submit_action_job",
        "get_execution_job",
        "collect_execution_job",
    ),
    "programmable_analysis": (
        "submit_analysis_program",
        "get_execution_job",
        "collect_execution_job",
    ),
}
_LEGACY_INVESTIGATION_JOB_TOOL_FAMILIES = {
    "transition_state_path": (
        "submit_transition_state_path_job",
        "get_transition_state_path_job",
        "collect_transition_state_path_job",
    ),
}
_SCIENTIFIC_CONTEXT_PAYLOAD_FIELDS = frozenset(
    {
        "hypothesis",
        "explicit_conditions",
        "frozen_context",
        "working_memory",
        "calculation_unit",
        "initial_transition_state_structures",
        "initial_calculation_structures",
        "constraint_semantics",
        "workspace_input_files",
        "guidance",
        "materials",
    }
)

_CASE3_WORKSPACE_INPUT_PROMPT = (
    "本次 Case 3 输入文件已经放在当前受限 workspace 的 inputs/case3/ 下。"
    "只使用 user JSON 给出的相对路径读取四个结构文件和二面角映射文件；"
    "请自行探索计算方法、工作流、扫描范围和步长，系统没有预先指定软件设置。"
).strip()

_TS_INITIAL_GUESS_GUIDANCE_PROMPT = (
    "user JSON 的 calculation_unit.transition_state_initial_guess_source 已声明时，"
    "必须使用该字段指定的来源，不得自行替换成其他初猜算法；当其值为 "
    "search_transition_state_path 时，必须实际调用 TS Pipeline 生成候选。该字段未声明时，"
    "才根据科学任务和可用工具选择过渡态（TS）初猜来源。TS Pipeline 或 NEB 的表观能量、表观手性、"
    "候选标签和几何外观都只是未验证线索，不能据此提前淘汰候选，也不能据此进行最终排序。"
    "TS Pipeline 返回且可读取的每一个候选初猜，都必须在相同条件下分别进行 Gaussian TS 优化和频率验证；"
    "在所有候选完成这些 Gaussian 验证之前，不得选择最终候选，也不得丢弃任一候选。"
    "在需要确认机制或反应端点身份时，还应继续进行双向 IRC。最终取舍只能依据实际 Gaussian "
    "收敛情况、与反应模式对应的虚频、IRC 连通性以及可比的能量或自由能。user JSON 提供的 "
    "initial_transition_state_structures（如有）也只是本轮的候选初猜，不是已经验证的结果，"
    "其中每个候选同样必须完成上述验证；不能因为初猜的标签、表观属性或几何外观而跳过后续验证。"
)
_INITIAL_CALCULATION_STRUCTURE_PROMPT = (
    "user JSON 中的 initial_calculation_structures 是受控初始几何，不是本次计算结果。"
    "提交 scan_reaction_coordinates 时，必须逐值保留 atomic_numbers、coordinates_angstrom、"
    "charge 和 multiplicity；coordinate_atom_indices 是输入记录所用的 1-based 索引，"
    "不得直接提交给 MiniChem。MiniChem action_settings.atom_indices 必须使用对应记录的"
    "tool_coordinate_atom_indices（0-based），steps 必须使用 scan.tool_step_count，value_unit 必须"
    "使用 scan.tool_value_unit。所有能量、曲线、排序和结论只能来自本次工具实际结果。"
)
_ADVISORY_CALCULATION_STRUCTURE_PROMPT = (
    "本次请求的 constraint_semantics.policy=advisory。user JSON 中的"
    " initial_calculation_structures 只把 atomic_numbers、coordinates_angstrom、charge、"
    "multiplicity、coordinate_atom_indices、tool_coordinate_atom_indices 和 atom_roles"
    "作为 immutable facts（不可变初始事实）；不得改变四个固定科学对象、对象数量、原子角色或坐标索引。"
    "其中 structure.scan 的 start_degrees、end_degrees、step_degrees、relaxed，以及"
    "calculation_unit.method_requirements 中的 method、basis、solvent、symmetry、范围和步长"
    "只是 advisory reference，不得把 advisory reference "
    "当作硬约束；可以采用或调整，但必须在最终 narrative/observations 中说明调整理由和"
    "对可比性的影响。coordinate_atom_indices 是 1-based 输入索引，不得直接提交给 MiniChem；"
    "MiniChem action_settings.atom_indices 必须使用对应记录的 tool_coordinate_atom_indices"
    "（0-based）。所有能量、曲线、排序和结论只能来自本次工具实际结果。"
)
_CASE3_ADVISORY_BASELINE_PROMPT = (
    "Case 3 的 advisory reference 不是硬约束。在工具支持且没有本次任务失败证据时，"
    "优先尝试作者基线；其中 DMA 的准确化学名称是 N,N-Dimethylacetamide，不得自行"
    "展开为另一种溶剂。作者扫描基线是从输入几何正向位移 60°，四套对象分别起步；"
    "若工具使用绝对"
    "start_value/end_value，应采用 constraint_semantics.advisory_references.relaxed_scan"
    ".absolute_tool_endpoints_degrees 中对应对象的端点，不要把 ortho 初始二面角强制改为 0°。"
    "该执行顺序是非强制性建议，遇到工具失败或科学证据时仍可调整，"
    "并说明调整理由及其对四个对象可比性的影响。"
)
_INITIAL_SYSTEM_PROMPT = """
根据 user JSON 中的科学问题、条件、假设、冻结上下文和计算单元开展调查。
只把实际工具返回的状态、结果和 artifact 当作科学证据；不要猜测字段含义、计算数值或路径内容。
所有提交的任务都使用同一 job_id 协议：提交后记住 job_id，继续查询同一个 job_id；Python
脚本类任务可以较快查询；Gaussian 或其他化学计算提交后约 1、5、10、20、30 分钟查询，之后每约 30 分钟持续查询。查询明确终态后再
collect 读取完整结果和 artifact；collect 前不要把任务当作已完成的证据。一次只推进一个
尚未 collect 的任务，完成收集后才开始下一项。

submit 在创建前明确失败时，根据真实错误修正后重试；已经取得 job_id 或提交结果不确定时，
不要重复提交同一任务。工具目录无法完成科学要求时，返回 tool_capability_gap 并说明缺少的能力。
完成所需计算和证据核对后，只返回符合 response schema 的最终 JSON；不需要计算时也直接返回最终 JSON。
""".strip()

_REPAIR_SYSTEM_PROMPT = """
根据 user JSON 中的 result_version 和 repair_advice 修正当前科学调查。
只把实际工具返回的状态、结果和 artifact 当作科学证据；不要猜测字段含义、计算数值或路径内容。
所有提交的任务都使用同一 job_id 协议：提交后记住 job_id，继续查询同一个 job_id；Python
脚本类任务可以较快查询；Gaussian 或其他化学计算提交后约 1、5、10、20、30 分钟查询，之后每约 30 分钟持续查询。查询明确终态后再
collect 读取完整结果和 artifact；collect 前不要把任务当作已完成的证据。一次只推进一个
尚未 collect 的任务，完成收集后才开始下一项。

submit 在创建前明确失败时，根据真实错误修正后重试；已经取得 job_id 或提交结果不确定时，
不要重复提交同一任务。工具目录无法完成科学要求时，返回 tool_capability_gap 并说明缺少的能力。
完成所需计算和证据核对后，只返回符合 response schema 的最终 JSON。
""".strip()

_JOB_WAKE_SYSTEM_PROMPT = """
这是一次外部中断后的恢复调用。围绕 user JSON 提供的同一 job_id 查询；Gaussian 或其他化学计算提交后约 1、5、10、20、30 分钟查询，之后每约 30 分钟持续查询；确认终态后 collect，
只把实际结果和 artifact 当作科学证据。不要重复提交已有 job_id，也不要把查询前的内容当作结果。
收集完成后继续必要的科学计算；工具目录无法完成要求时，返回 tool_capability_gap。完成证据核对后，
只返回符合 response schema 的最终 JSON。
""".strip()

_JOB_CONTINUATION_SYSTEM_PROMPT = """
这是一次 Investigation 的继续调用，可能发生在原作业仍在运行时，也可能发生在原作业已经结束后。
继续使用当前 scope 中已提交作业的身份，不要因为本次调用没有终态通知而重新提交。你可以随时调用
list_execution_jobs 查看当前 scope 中你提交过的所有作业（包括已完成作业和仍在运行的作业），再用
同一个 job_id 调用对应的状态和 collect 工具。Python 脚本类任务可以较快查询，Gaussian 或其他化学
计算提交后约 1、5、10、20、30 分钟查询，之后每约 30 分钟持续查询。无论历史 turn 或 list 是否已经显示作业结束，本次 continuation 的第一项
作业工具调用必须是对 waiting job_id 的对应状态工具 get；list 不算本次 get，历史 turn 中的 get
也不算本次证据。只有本次 turn 的 get 回执明确包含同一 job_id 的终态后才可以 collect，并且
collect 必须继续使用这个 job_id。失败的工具调用、get 或 collect 必须读取并利用返回的原始错误信息
来修正后续参数或调用方式。

本次 continuation 可能需要收集已有作业后继续下一项科学计算，也可能只需继续观察当前作业；不要把
查询前的猜测当作科学证据，不要重复提交已有 job_id 或提交结果不确定的任务。工具目录无法完成科学
要求时，返回 tool_capability_gap 并说明缺少的能力。完成所需计算和证据核对后，只返回符合 response
schema 的最终 JSON。
""".strip()

_TASK_COMPLETION_SYSTEM_PROMPT = """
提交的计算任务可能运行很长时间，尤其是 Gaussian。只要同一 job_id 仍处于排队或运行状态，
不得取消、中断、放弃或重新提交该任务，也不得因为等待时间长而提前结束调查；必须按约定的查询
节奏等待并持续查询同一 job_id，直到得到明确终态后再 collect。

只有在 user JSON 明确要求的全部必做实验、科学对象、候选、扫描、比较和验证项都取得满足任务要求、
可用于科学判断的结果后，本次调查才算完成。失败、未收敛、不完整、缺失或只有诊断信息的作业不算
完成；所有作业已进入终态或当前没有活动作业，也不代表科学任务已经完成。

只要仍有必做项未完成，就必须继续使用允许的工具修正、重试或采用科学上可比的替代方案，不得提交
表示任务完成的 InvestigationResult，也不得把“部分支持”“部分完成”或“未完整验证”包装成最终成功。
若经过多次合理的修正、重试或替代尝试后仍无法补齐必做项，本次调查必须以执行失败结束：可以报告
已完成项、未完成项、尝试过程和失败证据，但不得标记为成功，也不得提交 InvestigationResult。
工具目录确实缺少完成科学要求所必需的能力时，返回 tool_capability_gap 并说明缺少的能力。
""".strip()

_REVIEW_SYSTEM_PROMPT = """
你是任务级 Review 模型。审查 user JSON 中的全部 current_results 和
evidence_snapshot；每个结果由完整的
(hypothesis_execution_input_record_id, result_version) 配对确定，不要丢弃
输入中的结果，也不要只审查最近返修的结果。当前 user JSON 和 evidence_snapshot 是唯一可审查
事实，不能从历史对话补充、替代或合并结果。逐项独立检查回顾性计算过程、工具引用、观测、
限制和证据绑定，并为每个输入结果输出对应的 finding。

每项 validity 只能是 pass 或 fail。pass 时必须且只能给 hypothesis_judgment=supported
或 refuted；fail 时必须且只能给可执行的 repair_advice，不得给 hypothesis_judgment。
整体还必须回答 can_answer_original_question。若所有结果 validity 都是 pass、至少一项
finding 为 supported 且结果足以回答原科学问题，输出 true，并且只提供非空 final_answer；
answerability_explanation 必须为 null。final_answer 应组织输入结果后直接回答原问题，可以引用
公开科学来源。若需要引用来源，只能引用当前输入或 evidence_snapshot 已经提供的来源；没有
对应来源时不得凭模型记忆新增文献、链接或 DOI。
若所有结果 validity 都是 pass 且存在 supported、但结果仍不足以回答原问题，输出 false，并且
final_answer 必须为 null、answerability_explanation 必须为非空文本。这是唯一允许输出
answerability_explanation 非空文本的情形。任一项为 fail 时输出 false，并且只为对应 finding
提供 repair_advice；此时 final_answer 和 answerability_explanation 都必须为 null。若所有结果
validity 都是 pass 且全部 refuted，输出 false，findings 只保留各自的 hypothesis_judgment，
final_answer 和 answerability_explanation 都必须为 null，由领域状态设置 revise_hypothesis 标识。
不能添加其它中间状态或科学判断。无论哪一种分支，final_answer 与 answerability_explanation
两个字段都必须存在；不适用时返回 null，不要省略。

若任一 current_result 或其证据表明，原任务要求的必做实验、科学对象、候选、扫描、比较或验证项
存在缺失、失败、未收敛或不完整，或只有诊断信息而没有可用于科学判断的结果，对应 finding 的
validity 必须为 fail，并在 repair_advice 中明确指出必须补齐的项目和验收结果。此时必须输出
can_answer_original_question=false、final_answer=null、answerability_explanation=null。
不得把“部分支持”、其它子项成功、limitations、所有作业已进入终态或当前没有活动作业作为 pass
的理由；只要必做项未补齐，就必须要求 Investigation 继续完成实验。本请求不判断
或声称返修预算是否耗尽；外层确认耗尽后会发送独立的 task_finalization 请求。

evidence_snapshot 是输入结果的绑定只读快照；manifest 的 source_artifact_ref 是外层已核对的
MiniChem 原始产物身份，必须与对应 InvestigationResult.artifact_refs 一一对应；manifest 的
Evidence artifact_id 只用于 reader 读取，两个命名空间不能互换、不能把其中一个当作另一个。
如果结果摘要不足，可通过随请求绑定的只读 evidence reader 按 Evidence artifact_id 读取快照中的
证据。Review 只能读取这些证据，没有 MiniChem 提交权限，不得提交、推进或猜测任何计算，也不
要把未提供的工具结果当作事实。
只返回 response schema 规定的原始 JSON object，不输出 Markdown 或额外字段。
""".strip()

_TASK_FINALIZATION_SYSTEM_PROMPT = """
你是任务级最终成稿模型。外层已根据 TaskCreated 冻结事实确认返修预算耗尽。
本次不再执行 Review，不得输出 validity、repair_advice、RepairRequired 或任何继续调查建议。

只读取 user JSON 中的全部 current_results 和绑定 evidence_snapshot。为每个输入
hypothesis_execution_input_record_id 输出一条忠实的 summary，不得遗漏、合并或新增执行输入。然后直接形成
给研究员的 final_answer：现有证据足以回答原科学问题时，answerability 为 answerable；
现有证据不足时，answerability 为 unanswerable，final_answer 必须具体说明已知结果、
关键限制和为什么不能得出更强结论。两种情况都必须输出非空 final_answer。

不得凭模型记忆新增文献、链接、DOI、实验或计算结果。如需核对摘要中未展开的内容，
只能使用本请求绑定的只读 evidence reader；不得提交 MiniChem 作业。
只返回 response schema 规定的原始 JSON object，不输出 Markdown 或额外字段。
""".strip()


def _require_text(value: Any, label: str) -> str:
    """校验请求身份或短文本，允许公开来源 URL 但拒绝本机路径和凭据。"""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} 必须是非空字符串")
    _assert_no_path_or_secret(value, label=label, allow_public_source_uris=True)
    return value.strip()


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    """校验并复制一个结构化 object；列表和纯文本不能冒充上下文 object。"""

    if not isinstance(value, Mapping):
        raise TypeError(f"{label} 必须是 JSON object")
    normalized = _canonicalize_json_value(
        value,
        label=label,
        allow_public_source_uris=True,
    )
    if not isinstance(normalized, dict):  # pragma: no cover - 由上面的检查保证
        raise TypeError(f"{label} 必须是 JSON object")
    return normalized


def _require_scientific_mapping(value: Any, label: str) -> dict[str, Any]:
    """校验公开科学上下文 object，允许正文排版空白但保持身份引用严格。"""

    if not isinstance(value, Mapping):
        raise TypeError(f"{label} 必须是 JSON object")
    normalized = require_safe_public_scientific_context(value, label=label)
    if not isinstance(normalized, dict):  # pragma: no cover - 由上面的检查保证
        raise TypeError(f"{label} 必须是 JSON object")
    return normalized


def _select_investigation_system_prompt(
    base_prompt: str,
    initial_calculation_structures: Mapping[str, Any] | None = None,
    constraint_semantics: Mapping[str, Any] | None = None,
    workspace_input_files: Mapping[str, Any] | None = None,
) -> str:
    """为调查请求追加完成门槛、TS 规则以及可选的建议性约束说明。"""

    selected = (
        f"{base_prompt}\n\n{_TASK_COMPLETION_SYSTEM_PROMPT}"
        f"\n\n{_TS_INITIAL_GUESS_GUIDANCE_PROMPT}"
    )
    is_advisory = (
        isinstance(constraint_semantics, Mapping)
        and constraint_semantics.get("policy") == "advisory"
    )
    if is_advisory:
        selected = f"{selected}\n\n{_ADVISORY_CALCULATION_STRUCTURE_PROMPT}"
        if (
            constraint_semantics.get("schema")
            == _CASE3_ADVISORY_CONSTRAINT_SCHEMA
        ):
            selected = f"{selected}\n\n{_CASE3_ADVISORY_BASELINE_PROMPT}"
    elif initial_calculation_structures:
        selected = f"{selected}\n\n{_INITIAL_CALCULATION_STRUCTURE_PROMPT}"
    if workspace_input_files:
        selected = f"{selected}\n\n{_CASE3_WORKSPACE_INPUT_PROMPT}"
    return selected


def _select_confirmed_execution_system_prompt(base_prompt: str) -> str:
    """构造确认执行输入使用的通用 system prompt。

    确认后的执行输入不能携带旧 Case 的 TS、workspace 或 advisory 说明；这些
    内容若存在，必须作为结构化 user 字段显式传入。共享的调查完成门槛仍然保留。
    """

    return f"{base_prompt}\n\n{_TASK_COMPLETION_SYSTEM_PROMPT}"


def _optional_mapping(value: Any, label: str) -> dict[str, Any]:
    """把缺省上下文规范化为空 object。"""

    return {} if value is None else _require_mapping(value, label)


def _optional_scientific_mapping(value: Any, label: str) -> dict[str, Any]:
    """把缺省公开科学上下文规范化为空 object。"""

    return {} if value is None else _require_scientific_mapping(value, label)


def _optional_constraint_semantics(value: Any) -> dict[str, Any]:
    """校验可选 constraint semantics，并要求其明确声明 strict 或 advisory policy。"""

    normalized = _optional_scientific_mapping(value, "constraint_semantics")
    if normalized and normalized.get("policy") not in {"strict", "advisory"}:
        raise ValueError("constraint_semantics.policy 必须是 strict 或 advisory")
    return normalized


def _optional_workspace_input_files(value: Any) -> dict[str, Any]:
    """校验只含当前 workspace 相对路径的 Case 3 输入说明。"""

    if value is None or (isinstance(value, Mapping) and not value):
        return {}
    if not isinstance(value, Mapping) or set(value) != {
        "structure_files",
        "dihedral_mapping_file",
        "instructions",
    }:
        raise ValueError("workspace_input_files 字段必须完整且不能带外部路径")
    structure_files = value["structure_files"]
    if not isinstance(structure_files, Mapping) or set(structure_files) != {
        "origin",
        "ortho",
        "meta",
        "para",
    }:
        raise ValueError("workspace_input_files.structure_files 必须覆盖四种结构")
    normalized_files = {
        identity: structure_files[identity]
        for identity in ("origin", "ortho", "meta", "para")
    }
    for identity, path in normalized_files.items():
        expected = f"inputs/case3/{identity}.gjf"
        if path != expected:
            raise ValueError(
                "workspace_input_files 只能使用 inputs/case3 下的固定相对文件名"
            )
    if value["dihedral_mapping_file"] != "inputs/case3/dihedral_atoms.json":
        raise ValueError("workspace_input_files 的二面角映射路径非法")
    instructions = value["instructions"]
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("workspace_input_files.instructions 必须是非空文本")
    normalized = {
        "structure_files": normalized_files,
        "dihedral_mapping_file": value["dihedral_mapping_file"],
        "instructions": instructions,
    }
    return _require_scientific_mapping(normalized, "workspace_input_files")


def _optional_execution_guidance(value: Any) -> str | None:
    """校验确认后仅属于当前 Session 的可选 guidance。"""

    if value is None:
        return None
    return _require_text(value, "guidance")


def _normalize_material_descriptors(
    value: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """校验物化后的材料 descriptor，只保留模型可见安全字段。

    ``workspace_relpath`` 必须是物化 seam 生成的 POSIX 相对路径；本函数不读取
    文件，也不接受 ``content_ref``、绝对路径或其它 Working Memory 身份字段。
    """

    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("materials 必须是 object 数组")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        label = f"materials[{index}]"
        if not isinstance(item, Mapping):
            raise TypeError(f"{label} 必须是 JSON object")
        allowed = {
            "workspace_relpath",
            "filename",
            "media_type",
            "size_bytes",
            "sha256",
            "description",
        }
        if not set(item).issubset(allowed) or not {
            "workspace_relpath",
            "filename",
            "media_type",
            "size_bytes",
            "sha256",
        }.issubset(item):
            raise ValueError(f"{label} 字段合同不匹配")
        path = item["workspace_relpath"]
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or _MATERIAL_WINDOWS_ABSOLUTE_RE.match(path) is not None
            or "\\" in path
            or "://" in path
            or _MATERIAL_PATH_CONTROL_RE.search(path) is not None
        ):
            raise ValueError(f"{label}.workspace_relpath 必须是安全相对路径")
        pure_path = PurePosixPath(path)
        if path != pure_path.as_posix() or "." in pure_path.parts or ".." in pure_path.parts:
            raise ValueError(f"{label}.workspace_relpath 必须是规范相对路径")
        filename = item["filename"]
        if (
            not isinstance(filename, str)
            or not filename
            or filename in {".", ".."}
            or "/" in filename
            or "\\" in filename
            or "://" in filename
            or _MATERIAL_PATH_CONTROL_RE.search(filename) is not None
            or _MATERIAL_WINDOWS_ABSOLUTE_RE.match(filename) is not None
        ):
            raise ValueError(f"{label}.filename 必须是安全 basename")
        if pure_path.name != filename:
            raise ValueError(f"{label}.workspace_relpath 与 filename 不一致")
        media_type = item["media_type"]
        if (
            not isinstance(media_type, str)
            or not media_type.strip()
            or _MATERIAL_PATH_CONTROL_RE.search(media_type) is not None
            or media_type.startswith("/")
            or _MATERIAL_WINDOWS_ABSOLUTE_RE.match(media_type) is not None
            or "://" in media_type
        ):
            raise ValueError(f"{label}.media_type 必须是安全非空文本")
        size_bytes = item["size_bytes"]
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
        ):
            raise ValueError(f"{label}.size_bytes 必须是非负整数")
        sha256 = item["sha256"]
        if not isinstance(sha256, str) or _HEX_SHA256_RE.fullmatch(sha256) is None:
            raise ValueError(f"{label}.sha256 必须是 64 位小写 SHA-256")
        descriptor: dict[str, Any] = {
            "workspace_relpath": path,
            "filename": filename,
            "media_type": media_type.strip(),
            "size_bytes": size_bytes,
            "sha256": sha256,
        }
        description = item.get("description")
        if description is not None:
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f"{label}.description 必须是非空文本或 null")
            descriptor["description"] = _require_text(
                description,
                f"{label}.description",
            )
        normalized.append(descriptor)
    return normalized


def _optional_initial_transition_state_structures(
    value: Any,
) -> dict[str, Any]:
    """校验可选过渡态初始结构；空 object 表示普通任务没有该输入。"""

    if value is None or (isinstance(value, Mapping) and not value):
        return {}
    try:
        return normalize_initial_transition_state_structures(value)
    except ValueError as exc:
        raise ValueError("initial_transition_state_structures 合同非法") from exc


def _optional_initial_calculation_structures(value: Any) -> dict[str, Any]:
    """校验可选普通计算初始结构；空 object 表示没有该输入。"""

    if value is None or (isinstance(value, Mapping) and not value):
        return {}
    try:
        return normalize_initial_calculation_structures(value)
    except ValueError as exc:
        raise ValueError("initial_calculation_structures 合同非法") from exc


def _normalize_evidence_reader_tool(value: Mapping[str, Any]) -> dict[str, str]:
    """只接受 Review Evidence MCP 实际注册的唯一只读工具描述。"""

    normalized = _require_mapping(value, "evidence_reader_tool")
    expected = {
        "name": "read_snapshot_artifact",
        "kind": "readonly_evidence_reader",
    }
    if normalized != expected:
        raise ValueError(
            "evidence_reader_tool 必须精确描述只读 read_snapshot_artifact"
        )
    return expected


def _require_hypothesis(hypothesis: Mapping[str, Any]) -> dict[str, Any]:
    """要求调查假设以 object 提供，避免把未结构化文本直接塞入 prompt。"""

    return _require_mapping(hypothesis, "hypothesis")


def _normalize_text_sequence(value: Sequence[str] | None, label: str) -> list[str]:
    """规范化可选的公开工具名称列表，拒绝集合和路径。"""

    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{label} 必须是字符串数组")
    normalized = [_require_text(item, f"{label}[{index}]") for index, item in enumerate(value)]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} 不能包含重复项")
    return sorted(normalized)


def _normalize_tool_descriptors(
    value: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """规范化可选工具描述；只复制 JSON object，不接收本机路径。"""

    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("tool_descriptors 必须是 object 数组")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        normalized.append(_require_mapping(item, f"tool_descriptors[{index}]"))
    return sorted(
        normalized,
        key=lambda item: json.dumps(
            item,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _canonical_message_payload(payload: Mapping[str, Any]) -> str:
    """将已校验 payload 序列化为无空白、键排序的 user JSON。"""

    normalized: dict[str, Any] = {}
    if "scientific_question" in payload:
        normalized["scientific_question"] = require_safe_public_scientific_text(
            payload["scientific_question"],
            label="scientific_question",
        )
    for key, value in payload.items():
        if key == "scientific_question":
            continue
        if not isinstance(key, str) or not key.strip():
            raise ValueError("prompt payload 的 object key 必须是非空字符串")
        _assert_no_path_or_secret(key, label="prompt payload.<key>")
        if key == "guidance":
            normalized[key] = _optional_execution_guidance(value)
        elif key == "materials":
            normalized[key] = _normalize_material_descriptors(value)
        elif key in _SCIENTIFIC_CONTEXT_PAYLOAD_FIELDS:
            normalized[key] = require_safe_public_scientific_context(
                value,
                label=f"prompt payload.{key}",
            )
        else:
            normalized[key] = _canonicalize_json_value(
                value,
                label=f"prompt payload.{key}",
                allow_public_source_uris=True,
            )
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _request_envelope(
    *,
    model: str,
    role: str,
    stage: str,
    profile: str,
    system_prompt: str,
    payload: Mapping[str, Any],
    response_schema: Mapping[str, Any],
    metadata: Mapping[str, Any],
    thread_id: str | None = None,
    tools: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """组装 Harness request，保留外部 session 路由元数据而不写入 user facts。"""

    normalized_model = _require_text(model, "model")
    normalized_metadata = _require_mapping(metadata, "metadata")
    envelope: dict[str, Any] = {
        "model": normalized_model,
        "role": role,
        "stage": stage,
        "profile": profile,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _canonical_message_payload(payload)},
        ],
        "response_schema": dict(response_schema),
        "metadata": normalized_metadata,
    }
    normalized_tools = _normalize_tool_descriptors(tools)
    if normalized_tools:
        envelope["tools"] = normalized_tools
    if thread_id is not None:
        envelope["thread_id"] = _require_text(thread_id, "thread_id")
    return envelope


def build_initial_investigation_request(
    research_task_id: str,
    investigation_context_id: str,
    hypothesis_execution_input_record_id: str,
    scientific_question: str,
    hypothesis: Mapping[str, Any],
    *,
    explicit_conditions: Mapping[str, Any] | None = None,
    frozen_context: Mapping[str, Any] | None = None,
    working_memory: Mapping[str, Any] | None = None,
    calculation_unit: Mapping[str, Any] | None = None,
    initial_transition_state_structures: Mapping[str, Any] | None = None,
    initial_calculation_structures: Mapping[str, Any] | None = None,
    constraint_semantics: Mapping[str, Any] | None = None,
    workspace_input_files: Mapping[str, Any] | None = None,
    guidance: str | None = None,
    materials: Sequence[Mapping[str, Any]] | None = None,
    confirmed_execution_input: bool = False,
    available_tools: Sequence[str] | None = None,
    tool_descriptors: Sequence[Mapping[str, Any]] | None = None,
    codex_thread_id: str | None = None,
    model: str = _DEFAULT_INVESTIGATION_MODEL,
) -> dict[str, Any]:
    """构造首次 Investigation 请求。

    输入包含 research_task、Investigation Context、执行输入身份、科学问题、
    结构化假设和冻结上下文。返回值是
    无副作用的 Harness request，使用严格 InvestigationResponse schema；函数不会
    创建 workflow，也不保存 thread。非法非 JSON 值、本机路径、私有 URI、凭据或
    别名冲突会立即抛出 ``TypeError`` 或 ``ValueError``；公开 https/doi 来源可保留在
    冻结科学上下文中。``working_memory`` 仅为旧调用方保留的兼容参数，会经过校验
    但永远不会进入模型请求；跨批次科学经验只能由 Hypothesis regeneration 请求
    显式投影。
    """

    normalized_research_task_id = _require_text(
        research_task_id,
        "research_task_id",
    )
    normalized_investigation_context_id = _require_text(
        investigation_context_id,
        "investigation_context_id",
    )
    normalized_execution_input_id = _require_text(
        hypothesis_execution_input_record_id,
        "hypothesis_execution_input_record_id",
    )
    normalized_question = require_safe_public_scientific_text(
        scientific_question,
        label="scientific_question",
    )
    normalized_hypothesis = _require_scientific_mapping(hypothesis, "hypothesis")
    normalized_conditions = _optional_scientific_mapping(
        explicit_conditions,
        "explicit_conditions",
    )
    normalized_context = _optional_scientific_mapping(frozen_context, "frozen_context")
    # 兼容旧调用方的字段仍做严格校验，但 Working Memory 不是首次调查的模型上下文。
    # 这样 Controller 可以继续保存恢复/路由摘要，同时不会把跨批次经验传给 Investigation。
    _optional_scientific_mapping(working_memory, "working_memory")
    normalized_unit = _optional_scientific_mapping(calculation_unit, "calculation_unit")
    normalized_initial_structures = _optional_initial_transition_state_structures(
        initial_transition_state_structures
    )
    normalized_initial_calculation_structures = _optional_initial_calculation_structures(
        initial_calculation_structures
    )
    normalized_constraint_semantics = _optional_constraint_semantics(
        constraint_semantics
    )
    normalized_workspace_input_files = _optional_workspace_input_files(
        workspace_input_files
    )
    normalized_guidance = _optional_execution_guidance(guidance)
    normalized_materials = _normalize_material_descriptors(materials)
    is_confirmed_execution = bool(
        confirmed_execution_input or guidance is not None or materials is not None
    )
    normalized_available_tools = _normalize_text_sequence(available_tools, "available_tools")
    normalized_thread_id = (
        None
        if codex_thread_id is None
        else _require_text(codex_thread_id, "codex_thread_id")
    )

    payload: dict[str, Any] = {
        "scientific_question": normalized_question,
        "hypothesis": normalized_hypothesis,
    }
    if is_confirmed_execution:
        if normalized_guidance is not None:
            payload["guidance"] = normalized_guidance
        if normalized_materials:
            payload["materials"] = normalized_materials
    else:
        payload["explicit_conditions"] = normalized_conditions
        payload["frozen_context"] = normalized_context
        if normalized_unit:
            payload["calculation_unit"] = normalized_unit
        if normalized_initial_structures:
            payload["initial_transition_state_structures"] = normalized_initial_structures
        if normalized_initial_calculation_structures:
            payload["initial_calculation_structures"] = (
                normalized_initial_calculation_structures
            )
        if normalized_constraint_semantics:
            payload["constraint_semantics"] = normalized_constraint_semantics
        if normalized_workspace_input_files:
            payload["workspace_input_files"] = normalized_workspace_input_files
    metadata = {
        "request_kind": "initial_investigation",
        "research_task_id": normalized_research_task_id,
        "investigation_context_id": normalized_investigation_context_id,
        "hypothesis_execution_input_record_id": normalized_execution_input_id,
    }
    if normalized_thread_id is not None:
        metadata["codex_thread_id"] = normalized_thread_id
    return _request_envelope(
        model=model,
        role="investigation",
        stage="investigation_and_experiment",
        profile="investigation",
        system_prompt=(
            _select_confirmed_execution_system_prompt(_INITIAL_SYSTEM_PROMPT)
            if is_confirmed_execution
            else _select_investigation_system_prompt(
                _INITIAL_SYSTEM_PROMPT,
                normalized_initial_calculation_structures,
                normalized_constraint_semantics,
                normalized_workspace_input_files,
            )
        ),
        payload=payload,
        response_schema=build_investigation_provider_json_schema(),
        metadata=metadata,
        thread_id=normalized_thread_id,
        tools=tool_descriptors or (),
    )


def build_repair_investigation_request(
    research_task_id: str,
    investigation_context_id: str,
    hypothesis_execution_input_record_id: str,
    codex_thread_id: str,
    result_version: str,
    repair_advice: str,
    *,
    repair_count: int | None = None,
    constraint_semantics: Mapping[str, Any] | None = None,
    confirmed_execution_input: bool = False,
    model: str = _DEFAULT_INVESTIGATION_MODEL,
) -> dict[str, Any]:
    """构造同一 Codex thread 上的 DeepSeek 返修请求。

    user payload 严格只包含指定身份、一个 ``result_version`` 和 DeepSeek 的
    ``repair_advice``；原科学上下文由原 thread 保留，函数不接受或拼接其它结果。
    返回值携带原 thread id 供外层 session 路由，且继续使用 InvestigationResponse
    schema。``repair_count`` 仅作可选审计元数据，范围必须为 1 到 3。
    """

    normalized_research_task_id = _require_text(
        research_task_id,
        "research_task_id",
    )
    normalized_investigation_context_id = _require_text(
        investigation_context_id,
        "investigation_context_id",
    )
    normalized_execution_input_id = _require_text(
        hypothesis_execution_input_record_id,
        "hypothesis_execution_input_record_id",
    )
    normalized_thread_id = _require_text(codex_thread_id, "codex_thread_id")
    normalized_version = _require_text(result_version, "result_version")
    normalized_advice = _require_text(repair_advice, "repair_advice")
    normalized_constraint_semantics = _optional_constraint_semantics(
        constraint_semantics
    )
    if repair_count is not None:
        if isinstance(repair_count, bool) or not isinstance(repair_count, int):
            raise TypeError("repair_count 必须是整数")
        if repair_count not in {1, 2, 3}:
            raise ValueError("repair_count 必须位于 1 到 3 之间")

    payload = {
        "result_version": normalized_version,
        "repair_advice": normalized_advice,
    }
    if normalized_constraint_semantics:
        payload["constraint_semantics"] = normalized_constraint_semantics
    metadata: dict[str, Any] = {
        "request_kind": "deepseek_repair",
        "research_task_id": normalized_research_task_id,
        "investigation_context_id": normalized_investigation_context_id,
        "hypothesis_execution_input_record_id": normalized_execution_input_id,
        "codex_thread_id": normalized_thread_id,
        "result_version": normalized_version,
    }
    if repair_count is not None:
        metadata["repair_count"] = repair_count
    return _request_envelope(
        model=model,
        role="investigation",
        stage="investigation_and_experiment",
        profile="investigation",
        system_prompt=(
            _select_confirmed_execution_system_prompt(_REPAIR_SYSTEM_PROMPT)
            if confirmed_execution_input
            else _select_investigation_system_prompt(
                _REPAIR_SYSTEM_PROMPT,
                constraint_semantics=normalized_constraint_semantics,
            )
        ),
        payload=payload,
        response_schema=build_investigation_provider_json_schema(),
        metadata=metadata,
        thread_id=normalized_thread_id,
    )


def build_investigation_job_wake_request(
    research_task_id: str,
    investigation_context_id: str,
    hypothesis_execution_input_record_id: str,
    codex_thread_id: str,
    *,
    job_id: str,
    operation_id: str,
    submission_key: str,
    terminal_status: str,
    action_id: str | None = None,
    backend_id: str | None = None,
    job_family: str = "transition_state_path",
    job_type: str | None = None,
    runtime: str | None = None,
    software_id: str | None = None,
    operation_fingerprint: str | None = None,
    scope_id: str | None = None,
    terminal_message: str | None = None,
    scientific_question: str | None = None,
    hypothesis: Mapping[str, Any] | None = None,
    explicit_conditions: Mapping[str, Any] | None = None,
    frozen_context: Mapping[str, Any] | None = None,
    calculation_unit: Mapping[str, Any] | None = None,
    initial_transition_state_structures: Mapping[str, Any] | None = None,
    initial_calculation_structures: Mapping[str, Any] | None = None,
    constraint_semantics: Mapping[str, Any] | None = None,
    workspace_input_files: Mapping[str, Any] | None = None,
    guidance: str | None = None,
    materials: Sequence[Mapping[str, Any]] | None = None,
    confirmed_execution_input: bool = False,
    available_tools: Sequence[str] | None = None,
    tool_descriptors: Sequence[Mapping[str, Any]] | None = None,
    model: str = _DEFAULT_INVESTIGATION_MODEL,
) -> dict[str, Any]:
    """构造同一 Codex thread 的外部中断恢复请求。

    输入是外部确认的终态、提交时冻结的完整作业身份和可选的完整科学上下文；
    函数不读取计算 payload、不收集 artifact，也不启动模型。返回值强制携带原始 thread 与 job
    identity，供 runtime 拒绝换 thread、换 job 或把普通 repair 请求冒充 wake。终态
    只允许 ``succeeded``、``failed``、``cancelled``、``expired`` 和 ``unknown``；仍在
    排队或运行的作业不能唤醒模型。模型可见正文只给出 ``job_id`` 和终态；完整
    operation/submission 身份和诊断信息只留在内部 metadata。提供科学上下文时必须同时提供
    scientific_question 和 hypothesis，完整上下文会按
    与首次 Investigation 相同的公开字段进入 user JSON，供清空历史后的恢复 wake 使用。
    """

    normalized_research_task_id = _require_text(
        research_task_id,
        "research_task_id",
    )
    normalized_investigation_context_id = _require_text(
        investigation_context_id,
        "investigation_context_id",
    )
    normalized_execution_input_id = _require_text(
        hypothesis_execution_input_record_id,
        "hypothesis_execution_input_record_id",
    )
    normalized_thread_id = _require_text(codex_thread_id, "codex_thread_id")
    identity_values = {
        "job_id": _require_text(job_id, "job_id"),
        "operation_id": _require_text(operation_id, "operation_id"),
        "submission_key": _require_text(submission_key, "submission_key"),
    }
    if action_id is not None:
        identity_values["action_id"] = _require_text(action_id, "action_id")
    if backend_id is not None:
        identity_values["backend_id"] = _require_text(backend_id, "backend_id")
    if operation_fingerprint is not None:
        identity_values["operation_fingerprint"] = _require_text(
            operation_fingerprint,
            "operation_fingerprint",
        )
    family_tools = _INVESTIGATION_JOB_TOOL_FAMILIES.get(job_family)
    if family_tools is None:
        raise ValueError("job_family 不是受支持的异步作业 family")
    normalized_scope = None if scope_id is None else _require_text(scope_id, "scope_id")
    if job_family in {"native_execution", "action_execution", "programmable_analysis"}:
        normalized_job_type = _require_text(job_type, "job_type")
        normalized_runtime = _require_text(runtime, "runtime")
        normalized_software_id = (
            None if software_id is None else _require_text(software_id, "software_id")
        )
    else:
        normalized_job_type = normalized_runtime = normalized_software_id = None
    normalized_status = _require_text(terminal_status, "terminal_status").lower()
    if normalized_status not in {"succeeded", "failed", "cancelled", "expired", "unknown"}:
        raise ValueError("terminal_status 必须是外部确认的终态")
    if terminal_message is not None:
        _require_text(terminal_message, "terminal_message")
    requested_tools = family_tools if available_tools is None else available_tools
    normalized_tools = _normalize_text_sequence(requested_tools, "available_tools")
    legacy_tools = _LEGACY_INVESTIGATION_JOB_TOOL_FAMILIES.get(job_family)
    has_canonical_query_tools = (
        family_tools[1] in normalized_tools and family_tools[2] in normalized_tools
    )
    has_legacy_query_tools = bool(
        legacy_tools is not None
        and legacy_tools[1] in normalized_tools
        and legacy_tools[2] in normalized_tools
    )
    if not has_canonical_query_tools and not has_legacy_query_tools:
        raise ValueError("available_tools 必须包含当前作业 family 的 status/collect 工具")
    descriptor_values = _normalize_tool_descriptors(tool_descriptors)
    normalized_initial_structures = _optional_initial_transition_state_structures(
        initial_transition_state_structures
    )
    normalized_initial_calculation_structures = _optional_initial_calculation_structures(
        initial_calculation_structures
    )
    normalized_constraint_semantics = _optional_constraint_semantics(
        constraint_semantics
    )
    normalized_workspace_input_files = _optional_workspace_input_files(
        workspace_input_files
    )
    normalized_guidance = _optional_execution_guidance(guidance)
    normalized_materials = _normalize_material_descriptors(materials)
    is_confirmed_execution = bool(
        confirmed_execution_input or guidance is not None or materials is not None
    )
    payload: dict[str, Any] = {
        "job_id": identity_values["job_id"],
        "terminal_status": normalized_status,
    }
    context_values = (
        (scientific_question, hypothesis, guidance, materials)
        if is_confirmed_execution
        else (
            scientific_question,
            hypothesis,
            explicit_conditions,
            frozen_context,
            calculation_unit,
            initial_transition_state_structures,
            initial_calculation_structures,
            workspace_input_files,
        )
    )
    if any(value is not None for value in context_values):
        if scientific_question is None or hypothesis is None:
            raise ValueError(
                "完整 wake 科学上下文必须同时包含 scientific_question 和 hypothesis"
            )
        payload.update(
            {
                "scientific_question": require_safe_public_scientific_text(
                    scientific_question,
                    label="scientific_question",
                ),
                "hypothesis": _require_scientific_mapping(hypothesis, "hypothesis"),
            }
        )
        if is_confirmed_execution:
            if normalized_guidance is not None:
                payload["guidance"] = normalized_guidance
            if normalized_materials:
                payload["materials"] = normalized_materials
        else:
            payload["explicit_conditions"] = _optional_scientific_mapping(
                explicit_conditions,
                "explicit_conditions",
            )
            payload["frozen_context"] = _optional_scientific_mapping(
                frozen_context,
                "frozen_context",
            )
            normalized_wake_unit = _optional_scientific_mapping(
                calculation_unit,
                "calculation_unit",
            )
            if normalized_wake_unit:
                payload["calculation_unit"] = normalized_wake_unit
            if normalized_initial_structures:
                payload["initial_transition_state_structures"] = (
                    normalized_initial_structures
                )
            if normalized_initial_calculation_structures:
                payload["initial_calculation_structures"] = (
                    normalized_initial_calculation_structures
                )
            if normalized_constraint_semantics:
                payload["constraint_semantics"] = normalized_constraint_semantics
            if normalized_workspace_input_files:
                payload["workspace_input_files"] = normalized_workspace_input_files
    elif normalized_constraint_semantics:
        payload["constraint_semantics"] = normalized_constraint_semantics
    metadata: dict[str, Any] = {
        "request_kind": "investigation_job_wake",
        "research_task_id": normalized_research_task_id,
        "investigation_context_id": normalized_investigation_context_id,
        "hypothesis_execution_input_record_id": normalized_execution_input_id,
        "codex_thread_id": normalized_thread_id,
        **identity_values,
        "terminal_status": normalized_status,
        "job_family": job_family,
    }
    if normalized_job_type is not None:
        metadata["job_type"] = normalized_job_type
    if normalized_runtime is not None:
        metadata["runtime"] = normalized_runtime
    if normalized_software_id is not None:
        metadata["software_id"] = normalized_software_id
    if "operation_fingerprint" in identity_values:
        metadata["operation_fingerprint"] = identity_values["operation_fingerprint"]
    if normalized_scope is not None:
        metadata["scope_id"] = normalized_scope
    return _request_envelope(
        model=model,
        role="investigation",
        stage="investigation_and_experiment",
        profile="investigation",
        system_prompt=(
            _select_confirmed_execution_system_prompt(_JOB_WAKE_SYSTEM_PROMPT)
            if is_confirmed_execution
            else _select_investigation_system_prompt(
                _JOB_WAKE_SYSTEM_PROMPT,
                normalized_initial_calculation_structures,
                normalized_constraint_semantics,
                normalized_workspace_input_files,
            )
        ),
        payload=payload,
        response_schema=build_investigation_provider_json_schema(),
        metadata=metadata,
        thread_id=normalized_thread_id,
        tools=descriptor_values,
    )


def build_investigation_job_continuation_request(
    research_task_id: str,
    investigation_context_id: str,
    hypothesis_execution_input_record_id: str,
    codex_thread_id: str,
    *,
    job_id: str,
    operation_id: str,
    submission_key: str,
    action_id: str | None = None,
    backend_id: str | None = None,
    job_family: str = "transition_state_path",
    job_type: str | None = None,
    runtime: str | None = None,
    software_id: str | None = None,
    operation_fingerprint: str | None = None,
    scope_id: str | None = None,
    scientific_question: str | None = None,
    hypothesis: Mapping[str, Any] | None = None,
    explicit_conditions: Mapping[str, Any] | None = None,
    frozen_context: Mapping[str, Any] | None = None,
    calculation_unit: Mapping[str, Any] | None = None,
    initial_transition_state_structures: Mapping[str, Any] | None = None,
    initial_calculation_structures: Mapping[str, Any] | None = None,
    constraint_semantics: Mapping[str, Any] | None = None,
    workspace_input_files: Mapping[str, Any] | None = None,
    guidance: str | None = None,
    materials: Sequence[Mapping[str, Any]] | None = None,
    confirmed_execution_input: bool = False,
    available_tools: Sequence[str] | None = None,
    tool_descriptors: Sequence[Mapping[str, Any]] | None = None,
    model: str = _DEFAULT_INVESTIGATION_MODEL,
) -> dict[str, Any]:
    """构造可在任意作业阶段继续同一 Codex thread 的 Investigation 请求。

    输入是等待 sidecar 中保存的完整作业身份和可选的公开科学上下文；动作只构造
    一个不产生副作用的 request，不查询 Controller、MiniChem 或 provider。与旧 wake 请求不同，
    本请求不要求外部先确认终态，但模型必须在当前 continuation turn 先用同一 job_id 调用状态
    工具 get；历史 turn 或 list 的结果不能替代当前 turn 的 get 证据。所有作业身份仍绑定原
    job/operation/submission 三元组，供 runtime 防止换作业或重复提交。
    """

    requested_tools = tuple(
        _INVESTIGATION_JOB_TOOL_FAMILIES.get(job_family, ())
        if available_tools is None
        else available_tools
    )
    if "list_execution_jobs" not in requested_tools:
        requested_tools = (*requested_tools, "list_execution_jobs")
    request = build_investigation_job_wake_request(
        research_task_id,
        investigation_context_id,
        hypothesis_execution_input_record_id,
        codex_thread_id,
        job_id=job_id,
        operation_id=operation_id,
        submission_key=submission_key,
        action_id=action_id,
        backend_id=backend_id,
        terminal_status="unknown",
        job_family=job_family,
        job_type=job_type,
        runtime=runtime,
        software_id=software_id,
        operation_fingerprint=operation_fingerprint,
        scope_id=scope_id,
        scientific_question=scientific_question,
        hypothesis=hypothesis,
        explicit_conditions=explicit_conditions,
        frozen_context=frozen_context,
        calculation_unit=calculation_unit,
        initial_transition_state_structures=initial_transition_state_structures,
        initial_calculation_structures=initial_calculation_structures,
        constraint_semantics=constraint_semantics,
        workspace_input_files=workspace_input_files,
        guidance=guidance,
        materials=materials,
        confirmed_execution_input=confirmed_execution_input,
        available_tools=requested_tools,
        tool_descriptors=tool_descriptors,
        model=model,
    )
    continuation = copy.deepcopy(request)
    messages = continuation.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("continuation request 缺少规范 messages")
    system_message = messages[0]
    user_message = messages[-1]
    if not isinstance(system_message, dict) or not isinstance(user_message, dict):
        raise ValueError("continuation request messages 结构非法")
    is_confirmed_execution = bool(
        confirmed_execution_input or guidance is not None or materials is not None
    )
    system_message["content"] = (
        _select_confirmed_execution_system_prompt(_JOB_CONTINUATION_SYSTEM_PROMPT)
        if is_confirmed_execution
        else _select_investigation_system_prompt(
            _JOB_CONTINUATION_SYSTEM_PROMPT,
            _optional_initial_calculation_structures(initial_calculation_structures),
            _optional_constraint_semantics(constraint_semantics),
            _optional_workspace_input_files(workspace_input_files),
        )
    )
    raw_payload = user_message.get("content")
    if not isinstance(raw_payload, str):
        raise ValueError("continuation request user payload 不是 JSON 文本")
    payload = json.loads(raw_payload)
    if not isinstance(payload, dict):
        raise ValueError("continuation request user payload 必须是 JSON object")
    payload.pop("terminal_status", None)
    user_message["content"] = _canonical_message_payload(payload)
    metadata = continuation.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("continuation request 缺少 metadata")
    metadata.pop("terminal_status", None)
    metadata["request_kind"] = "investigation_job_continuation"
    return continuation


_PUBLIC_RESULT_FIELDS = (
    "research_task_id",
    "investigation_context_id",
    "hypothesis_execution_input_record_id",
    "result_version",
    "self_judgment",
    "narrative",
    "job_refs",
    "operation_refs",
    "artifact_refs",
    "observations",
    "limitations",
)
_REQUIRED_RESULT_FIELDS = frozenset(
    {
        "hypothesis_execution_input_record_id",
        "result_version",
        "self_judgment",
        "narrative",
        "job_refs",
        "operation_refs",
        "artifact_refs",
        "observations",
        "limitations",
    }
)
_PRIVATE_RESULT_FIELDS = frozenset(
    {"codex_thread_id", "scope_id", "turn_id", "workspace_ref"}
)
_EVIDENCE_SNAPSHOT_FIELDS = frozenset(
    {
        "snapshot_id",
        "research_task_id",
        "investigation_context_id",
        "result_pairs",
    }
)
_EVIDENCE_PAIR_FIELDS = frozenset(
    {
        "hypothesis_execution_input_record_id",
        "result_version",
        "artifacts",
    }
)
_EVIDENCE_ARTIFACT_FIELDS = frozenset(
    {"artifact_id", "source_artifact_ref", "size", "sha256", "media_type"}
)


def _result_mapping(value: Any, index: int) -> dict[str, Any]:
    """校验并复制一份完整当前结果，保留 Review 所需的科学字段。"""

    if hasattr(value, "model_dump") and callable(value.model_dump):
        raw = value.model_dump(mode="json")
    elif isinstance(value, Mapping):
        raw = value
    else:
        raise TypeError(
            f"current_results[{index}] 必须是 InvestigationResult 或 JSON object"
        )
    raw = _canonicalize_json_value(
        raw,
        label=f"current_results[{index}]",
        allow_public_source_uris=True,
    )
    if not isinstance(raw, dict):
        raise TypeError(f"current_results[{index}] 必须是 JSON object")
    unknown = set(raw) - set(_PUBLIC_RESULT_FIELDS) - _PRIVATE_RESULT_FIELDS
    if unknown:
        raise ValueError(f"current_results[{index}] 含未注册字段: {sorted(unknown)}")
    missing = _REQUIRED_RESULT_FIELDS - set(raw)
    if missing:
        raise ValueError(
            f"current_results[{index}] 缺少完整结果字段: {sorted(missing)}"
        )
    execution_input_id = _require_text(
        raw["hypothesis_execution_input_record_id"],
        f"current_results[{index}].hypothesis_execution_input_record_id",
    )
    result_version = _require_text(
        raw["result_version"],
        f"current_results[{index}].result_version",
    )
    response_payload = {
        field: raw[field]
        for field in (
            "self_judgment",
            "narrative",
            "job_refs",
            "operation_refs",
            "artifact_refs",
            "observations",
            "limitations",
        )
    }
    response = validate_investigation_response(response_payload)
    selected: dict[str, Any] = {
        "hypothesis_execution_input_record_id": execution_input_id,
        "result_version": result_version,
        **response.model_dump(mode="json"),
    }
    for field in ("research_task_id", "investigation_context_id"):
        if field in raw:
            selected[field] = _require_text(
                raw[field],
                f"current_results[{index}].{field}",
            )
    # 按固定字段顺序重建 object，避免调用方插入顺序影响 prompt 指纹。
    return {key: selected[key] for key in _PUBLIC_RESULT_FIELDS if key in selected}


def _normalize_review_results(
    current_results: Sequence[Any],
    *,
    research_task_id: str,
    investigation_context_id: str,
) -> list[dict[str, Any]]:
    """执行任务级当前 Context barrier，按执行输入身份稳定排序结果。"""

    if isinstance(current_results, (str, bytes, bytearray)) or not isinstance(
        current_results,
        Sequence,
    ):
        raise TypeError("current_results 必须是结果 object 数组")
    if not 1 <= len(current_results) <= 5:
        raise ValueError("Review 请求必须一次读取一至五项当前结果")
    normalized_research_task_id = _require_text(
        research_task_id,
        "research_task_id",
    )
    normalized_context_id = _require_text(
        investigation_context_id,
        "investigation_context_id",
    )
    projected = [
        _result_mapping(item, index)
        for index, item in enumerate(current_results)
    ]
    pairs = [
        (
            item["hypothesis_execution_input_record_id"],
            item["result_version"],
        )
        for item in projected
    ]
    if len(set(pairs)) != len(pairs):
        raise ValueError(
            "Review 的 (hypothesis_execution_input_record_id, result_version) "
            "配对必须唯一"
        )
    execution_input_ids = [
        item["hypothesis_execution_input_record_id"] for item in projected
    ]
    if len(set(execution_input_ids)) != len(execution_input_ids):
        raise ValueError("Review 的 hypothesis_execution_input_record_id 必须唯一")
    for index, item in enumerate(projected):
        item_task_id = item.get("research_task_id")
        item_context_id = item.get("investigation_context_id")
        if (
            item_task_id is not None
            and item_task_id != normalized_research_task_id
        ):
            raise ValueError(
                f"current_results[{index}].research_task_id 与请求任务不一致"
            )
        if item_context_id is not None and item_context_id != normalized_context_id:
            raise ValueError(
                f"current_results[{index}].investigation_context_id 与请求 Context 不一致"
            )
        item["research_task_id"] = normalized_research_task_id
        item["investigation_context_id"] = normalized_context_id
    return sorted(
        projected,
        key=lambda item: item["hypothesis_execution_input_record_id"],
    )


def _read_evidence_snapshot_header(
    evidence_snapshot: Mapping[str, Any],
    *,
    research_task_id: str,
    investigation_context_id: str,
) -> tuple[str, list[Any]]:
    """校验证据快照的身份、字段集合和当前 Context 配对容器。"""

    snapshot = _require_mapping(evidence_snapshot, "evidence_snapshot")
    unknown = set(snapshot) - _EVIDENCE_SNAPSHOT_FIELDS
    if unknown:
        raise ValueError(f"evidence_snapshot 含未注册字段: {sorted(unknown)}")
    missing = _EVIDENCE_SNAPSHOT_FIELDS - set(snapshot)
    if missing:
        raise ValueError(f"evidence_snapshot 缺少字段: {sorted(missing)}")
    snapshot_id = _require_text(snapshot["snapshot_id"], "evidence_snapshot.snapshot_id")
    snapshot_task_id = _require_text(
        snapshot["research_task_id"],
        "evidence_snapshot.research_task_id",
    )
    snapshot_context_id = _require_text(
        snapshot["investigation_context_id"],
        "evidence_snapshot.investigation_context_id",
    )
    if (
        snapshot_task_id != research_task_id
        or snapshot_context_id != investigation_context_id
    ):
        raise ValueError("evidence_snapshot 的任务或 Context 身份不一致")
    raw_pairs = snapshot["result_pairs"]
    if isinstance(raw_pairs, (str, bytes, bytearray)) or not isinstance(raw_pairs, Sequence):
        raise TypeError("evidence_snapshot.result_pairs 必须是 object 数组")
    if not 1 <= len(raw_pairs) <= 5:
        raise ValueError("evidence_snapshot 必须包含一至五个结果配对")
    return snapshot_id, list(raw_pairs)


def _normalize_evidence_artifact(
    raw_artifact: Any,
    *,
    label: str,
) -> dict[str, Any]:
    """校验一个 Evidence artifact 的身份、哈希、大小和媒体类型。"""

    artifact = _require_mapping(raw_artifact, label)
    unknown = set(artifact) - _EVIDENCE_ARTIFACT_FIELDS
    if unknown:
        raise ValueError(f"{label} 含未注册字段: {sorted(unknown)}")
    missing = _EVIDENCE_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"{label} 缺少字段: {sorted(missing)}")

    artifact_id = artifact["artifact_id"]
    if not isinstance(artifact_id, str) or not artifact_id.strip():
        raise ValueError(f"{label} artifact_id 必须是非空字符串")
    _assert_no_path_or_secret(artifact_id, label=f"{label}.artifact_id")
    source_artifact_ref = artifact["source_artifact_ref"]
    if not isinstance(source_artifact_ref, str) or not source_artifact_ref.strip():
        raise ValueError(f"{label} source_artifact_ref 必须是非空字符串")
    _assert_no_path_or_secret(
        source_artifact_ref,
        label=f"{label}.source_artifact_ref",
    )
    normalized_source_ref = source_artifact_ref.strip()
    if artifact_id.strip() == normalized_source_ref:
        raise ValueError(f"{label} source_artifact_ref 与 Evidence artifact_id 不能相同")

    size = artifact["size"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise TypeError(f"{label} size 必须是非负整数")
    sha256 = artifact["sha256"]
    if not isinstance(sha256, str) or not _HEX_SHA256_RE.fullmatch(sha256.strip()):
        raise ValueError(f"{label} sha256 必须是 64 位小写 SHA-256")
    media_type = artifact["media_type"]
    if not isinstance(media_type, str) or not media_type.strip():
        raise ValueError(f"{label} media_type 必须是非空字符串")
    _assert_no_path_or_secret(media_type, label=f"{label}.media_type")
    return {
        "artifact_id": artifact_id.strip(),
        "source_artifact_ref": normalized_source_ref,
        "size": size,
        "sha256": sha256.strip(),
        "media_type": media_type.strip(),
    }


def _normalize_evidence_pair(
    raw_pair: Any,
    *,
    index: int,
) -> tuple[dict[str, Any], set[str], set[str]]:
    """校验一个结果配对并返回规范化 artifact 集合。"""

    pair_label = f"evidence_snapshot.result_pairs[{index}]"
    pair = _require_mapping(raw_pair, pair_label)
    unknown = set(pair) - _EVIDENCE_PAIR_FIELDS
    if unknown:
        raise ValueError(f"{pair_label} 含未注册字段: {sorted(unknown)}")
    for field in ("hypothesis_execution_input_record_id", "result_version"):
        if field not in pair:
            raise ValueError(f"{pair_label} 缺少 {field}")
    normalized_pair = {
        "hypothesis_execution_input_record_id": _require_text(
            pair["hypothesis_execution_input_record_id"],
            f"{pair_label}.hypothesis_execution_input_record_id",
        ),
        "result_version": _require_text(
            pair["result_version"],
            f"{pair_label}.result_version",
        ),
    }
    if "artifacts" not in pair:
        raise ValueError(f"{pair_label} 缺少 artifacts")
    raw_artifacts = pair["artifacts"]
    if isinstance(raw_artifacts, (str, bytes, bytearray)) or not isinstance(
        raw_artifacts,
        Sequence,
    ):
        raise TypeError(f"{pair_label}.artifacts 必须是 object 数组")
    normalized_artifacts = [
        _normalize_evidence_artifact(
            raw_artifact,
            label=f"{pair_label}.artifacts[{artifact_index}]",
        )
        for artifact_index, raw_artifact in enumerate(raw_artifacts)
    ]
    artifact_ids = {item["artifact_id"] for item in normalized_artifacts}
    if len(artifact_ids) != len(normalized_artifacts):
        raise ValueError(f"{pair_label}.artifacts 不能重复 artifact_id")
    source_refs = {item["source_artifact_ref"] for item in normalized_artifacts}
    if len(source_refs) != len(normalized_artifacts):
        raise ValueError(f"{pair_label}.artifacts 不能重复 source_artifact_ref")
    normalized_artifacts.sort(key=lambda item: item["artifact_id"])
    normalized_pair["artifacts"] = normalized_artifacts
    return normalized_pair, artifact_ids, source_refs


def _validate_evidence_pair_namespaces(
    index: int,
    artifact_ids: set[str],
    source_refs: set[str],
    seen_evidence_ids: set[str],
    seen_source_refs: set[str],
) -> None:
    """拒绝 Evidence 和 MiniChem artifact 标识在快照内重复或串命名空间。"""

    label = f"evidence_snapshot.result_pairs[{index}].artifacts"
    if artifact_ids & seen_evidence_ids:
        raise ValueError(f"{label} 不能跨 pair 复用 Evidence artifact_id")
    if source_refs & seen_source_refs:
        raise ValueError(f"{label} 不能跨 pair 复用 source_artifact_ref")
    if (
        artifact_ids & seen_source_refs
        or source_refs & seen_evidence_ids
        or artifact_ids & source_refs
    ):
        raise ValueError(f"{label} 不能跨命名空间复用 ID")


def _validate_evidence_pair_source_binding(
    pair: Mapping[str, Any],
    *,
    index: int,
    result_source_artifacts: Mapping[str, Sequence[str]],
    source_refs: set[str],
) -> None:
    """核对 Evidence pair 的 MiniChem 来源引用与当前结果完全一致。"""

    execution_input_id = pair["hypothesis_execution_input_record_id"]
    label = f"evidence_snapshot.result_pairs[{index}]"
    if execution_input_id not in result_source_artifacts:
        raise ValueError(
            f"{label} 的 hypothesis_execution_input_record_id 不在 current_results"
        )
    expected_source_refs = set(result_source_artifacts[execution_input_id])
    if source_refs != expected_source_refs:
        raise ValueError(
            f"{label}.source_artifact_ref 必须匹配当前结果的 artifact_refs"
        )


def _normalize_evidence_snapshot(
    evidence_snapshot: Mapping[str, Any],
    *,
    research_task_id: str,
    investigation_context_id: str,
    result_pairs: Sequence[tuple[str, str]],
    result_source_artifacts: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """校验不可变证据 manifest，并绑定两种独立 artifact 命名空间。

    当前结果的 ``artifact_refs`` 是外层 authenticated MiniChem verifier 已核对
    的原始科学产物身份；manifest 的 ``source_artifact_ref`` 必须逐 pair 完整
    对齐这些身份，而 manifest 的 ``artifact_id`` 只是 Review reader 内容绑定
    ID。函数绝不把两种 ID 直接比较或互相替换，并拒绝同一 snapshot 内任一
    命名空间的重复/跨 pair 复用。
    """

    snapshot_id, raw_pairs = _read_evidence_snapshot_header(
        evidence_snapshot,
        research_task_id=research_task_id,
        investigation_context_id=investigation_context_id,
    )
    normalized_pairs: list[dict[str, Any]] = []
    seen_evidence_ids: set[str] = set()
    seen_source_refs: set[str] = set()
    for index, item in enumerate(raw_pairs):
        normalized_pair, artifact_ids, source_refs = _normalize_evidence_pair(
            item,
            index=index,
        )
        _validate_evidence_pair_namespaces(
            index,
            artifact_ids,
            source_refs,
            seen_evidence_ids,
            seen_source_refs,
        )
        _validate_evidence_pair_source_binding(
            normalized_pair,
            index=index,
            result_source_artifacts=result_source_artifacts,
            source_refs=source_refs,
        )
        seen_evidence_ids.update(artifact_ids)
        seen_source_refs.update(source_refs)
        normalized_pairs.append(normalized_pair)
    snapshot_pairs = {
        (
            item["hypothesis_execution_input_record_id"],
            item["result_version"],
        )
        for item in normalized_pairs
    }
    if len(snapshot_pairs) != len(result_pairs) or snapshot_pairs != set(result_pairs):
        raise ValueError("evidence_snapshot 的配对必须与 current_results 完全一致")
    normalized_pairs.sort(
        key=lambda item: item["hypothesis_execution_input_record_id"]
    )
    return {
        "snapshot_id": snapshot_id,
        "research_task_id": research_task_id,
        "investigation_context_id": investigation_context_id,
        "result_pairs": normalized_pairs,
    }


def _build_task_result_request(
    research_task_id: str,
    investigation_context_id: str,
    scientific_question: str,
    current_results: Sequence[Any],
    evidence_snapshot: Mapping[str, Any],
    *,
    request_kind: str,
    system_prompt: str,
    response_schema: Mapping[str, Any],
    evidence_reader_tool: Mapping[str, Any] | None = None,
    review_context: Mapping[str, Any] | None = None,
    model: str = _DEFAULT_REVIEW_MODEL,
    review_thread_id: str | None = None,
) -> dict[str, Any]:
    """构造共用当前结果和 Evidence 快照的任务级模型请求。

    函数要求全部当前结果以唯一的
    (hypothesis_execution_input_record_id, result_version) 配对作为完整屏障；
    结果版本字符串在不同执行输入间可以相同。结果按执行输入身份排序，
    以便同一快照在恢复时得到相同请求指纹。evidence_snapshot 必须与这些配对
    及其 artifacts manifest 完全一致，作为只读证据读取器的绑定索引；可选的
    evidence_reader_tool 只
    描述这个读取工具，不授予 MiniChem 提交权限。函数不会暴露调查 thread、scope、
    turn 或本机路径。
    """

    if request_kind not in {"task_review", "task_finalization"}:
        raise ValueError("request_kind 未注册")
    normalized_research_task_id = _require_text(
        research_task_id,
        "research_task_id",
    )
    normalized_investigation_context_id = _require_text(
        investigation_context_id,
        "investigation_context_id",
    )
    normalized_question = require_safe_public_scientific_text(
        scientific_question,
        label="scientific_question",
    )
    normalized_results = _normalize_review_results(
        current_results,
        research_task_id=normalized_research_task_id,
        investigation_context_id=normalized_investigation_context_id,
    )
    result_pairs = [
        (
            item["hypothesis_execution_input_record_id"],
            item["result_version"],
        )
        for item in normalized_results
    ]
    result_source_artifacts = {
        item["hypothesis_execution_input_record_id"]: item["artifact_refs"]
        for item in normalized_results
    }
    normalized_snapshot = _normalize_evidence_snapshot(
        evidence_snapshot,
        research_task_id=normalized_research_task_id,
        investigation_context_id=normalized_investigation_context_id,
        result_pairs=result_pairs,
        result_source_artifacts=result_source_artifacts,
    )
    # 兼容旧调用方的 review_context 仍做输入校验，但 Review 模型只接收当前结果
    # 结果和绑定 evidence snapshot，不能把任意历史或 Working Memory 资料带入 user JSON。
    _optional_mapping(review_context, "review_context")
    normalized_thread_id = (
        None
        if review_thread_id is None
        else _require_text(review_thread_id, "review_thread_id")
    )

    payload = {
        "request_kind": request_kind,
        "research_task_id": normalized_research_task_id,
        "investigation_context_id": normalized_investigation_context_id,
        "scientific_question": normalized_question,
        "current_results": normalized_results,
        "evidence_snapshot": normalized_snapshot,
    }
    metadata: dict[str, Any] = {
        "request_kind": request_kind,
        "research_task_id": normalized_research_task_id,
        "investigation_context_id": normalized_investigation_context_id,
        "result_pairs": [list(pair) for pair in result_pairs],
        "evidence_snapshot_id": normalized_snapshot["snapshot_id"],
    }
    if normalized_thread_id is not None:
        metadata["review_thread_id"] = normalized_thread_id
    normalized_tools = (
        ()
        if evidence_reader_tool is None
        else (_normalize_evidence_reader_tool(evidence_reader_tool),)
    )
    return _request_envelope(
        model=model,
        role="task_review",
        stage=request_kind,
        profile="review",
        system_prompt=system_prompt,
        payload=payload,
        response_schema=response_schema,
        metadata=metadata,
        thread_id=normalized_thread_id,
        tools=normalized_tools,
    )


def build_task_review_request(
    research_task_id: str,
    investigation_context_id: str,
    scientific_question: str,
    current_results: Sequence[Any],
    evidence_snapshot: Mapping[str, Any],
    *,
    evidence_reader_tool: Mapping[str, Any] | None = None,
    review_context: Mapping[str, Any] | None = None,
    model: str = _DEFAULT_REVIEW_MODEL,
    review_thread_id: str | None = None,
) -> dict[str, Any]:
    """构造可要求定向返修的任务级 DeepSeek Review 请求。"""

    return _build_task_result_request(
        research_task_id,
        investigation_context_id,
        scientific_question,
        current_results,
        evidence_snapshot,
        request_kind="task_review",
        system_prompt=_REVIEW_SYSTEM_PROMPT,
        response_schema=build_task_review_provider_json_schema(),
        evidence_reader_tool=evidence_reader_tool,
        review_context=review_context,
        model=model,
        review_thread_id=review_thread_id,
    )


def build_task_finalization_request(
    research_task_id: str,
    investigation_context_id: str,
    scientific_question: str,
    current_results: Sequence[Any],
    evidence_snapshot: Mapping[str, Any],
    *,
    evidence_reader_tool: Mapping[str, Any] | None = None,
    review_context: Mapping[str, Any] | None = None,
    model: str = _DEFAULT_REVIEW_MODEL,
    review_thread_id: str | None = None,
) -> dict[str, Any]:
    """构造返修预算耗尽后不再要求修复的终局成稿请求。"""

    return _build_task_result_request(
        research_task_id,
        investigation_context_id,
        scientific_question,
        current_results,
        evidence_snapshot,
        request_kind="task_finalization",
        system_prompt=_TASK_FINALIZATION_SYSTEM_PROMPT,
        response_schema=build_task_finalization_provider_json_schema(),
        evidence_reader_tool=evidence_reader_tool,
        review_context=review_context,
        model=model,
        review_thread_id=review_thread_id,
    )
