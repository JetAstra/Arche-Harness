#!/usr/bin/env python3
"""通过 MiniChem MCP 验证论文 Case 1 的工具路径和输入缺口。

脚本先读取论文附录和 Case 1 结构化结果，保存静态步骤映射；服务可用时冻结
在线目录并检查 Action/open_execution 合同。Gaussian 回放使用附录公开的两条
TS final coordinates 生成论文精确 route deck，经 MCP native preflight 后才允许
提交；Step 7/10 只在本次 Step 4 终态 checkpoint 收集后串联提交，Step 11 只读取
本次 native 日志；已存在 operation 只查询同一 job，不自动重复提交。脚本明确区分
公开坐标回放、工具能力验证和论文从 SMILES 独立发现 TS 的科学结论。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

from chemistry_multiagent.tools.minichem_adapter import MiniChemMcpAdapter


ROOT = Path(__file__).resolve().parents[2]
CASE1_PATH = ROOT / "supplementary" / "case1.json"
DEFAULT_OUTPUT = (
    ROOT / "local_data" / "runtime" / "case1_harness" / "tool_validation"
)
# Case 1 的 native Gaussian 作业独占整台验证节点，避免多个小作业长期排队。
CASE1_NATIVE_CPU_CORES = 60
CASE1_NATIVE_MEMORY_MB = 320000
CASE1_NATIVE_GPU_COUNT = 0

_NATIVE_ACTIVE_STATUSES = frozenset(
    {"preparing", "validating", "submitting", "waiting", "in_progress", "running"}
)
_NATIVE_TERMINAL_FAILURE_STATUSES = frozenset(
    {"failed", "timeout", "timed_out", "cancelled", "canceled"}
)

CASE1_REACTANTS = {
    "p_nitrobenzaldehyde": "O=Cc1ccc(cc1)[N+](=O)[O-]",
    "acetone": "CC(=O)C",
    "triflic_acid": "C(F)(F)(F)S(=O)(=O)O",
    "chiral_diamine": "N1CCC[C@H]1C(N2CCCC2)",
    "product": "CC(=O)C(O)c1ccc([N+](=O)[O-])cc1",
}

ATOMIC_SYMBOLS = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    16: "S",
}

STATIC_STEPS = [
    {
        "step": 1,
        "paper_tool": "TSPipline",
        "candidate_actions": ["search_transition_state_path"],
        "input": "固定 Case 1 Action 无科学输入；resource_limits.gpu_count 至少为 1",
        "output": "候选反应路径及受控 ArtifactRef 交接",
        "execution": "not_executed",
        "gap": "该 Action 固定服务端科学流程；输出不是频率或双向 IRC 验证结果。",
    },
    {
        "step": 2,
        "paper_tool": "GenerateGaussianCode",
        "candidate_actions": ["generate_gaussian_code"],
        "input": "TS 优化与频率计算",
        "output": "Gaussian route section",
        "execution": "not_executed",
        "gap": "catalog 已提供作者原名 Action；静态回放仍使用论文精确 route，避免把一次模型生成结果冒充论文原始关键词。",
    },
    {
        "step": 3,
        "paper_tool": "xyz_to_gjf",
        "candidate_actions": ["xyz_to_gjf"],
        "input": "TS XYZ 和 Step 2 route",
        "output": "TS 优化 GJF",
        "execution": "not_executed",
        "gap": "catalog 已提供作者原名 Action；静态回放仍保留确定性 deck builder，以核对论文精确 route 与公开坐标。",
    },
    {
        "step": 4,
        "paper_tool": "Gaussian",
        "candidate_actions": ["optimize_geometry", "calculate_hessian"],
        "input": "两个 TS GJF；B3LYP/6-31G(d), opt=(ts,noeigentest,calcall,maxstep=5), freq, IEFPCM acetone, 303.15 K",
        "output": "两个优化 TS、频率（每个一个虚频）和能量",
        "execution": "not_executed",
        "gap": "论文原始 Step 1 初猜没有公开；本验证可用 case1.json 的 final coordinates 做 native route 回放，但不能据此证明从 SMILES 独立发现 TS。",
    },
    {
        "step": 5,
        "paper_tool": "generate_gaussian_code",
        "candidate_actions": ["generate_gaussian_code"],
        "input": "IRC calculation",
        "output": "IRC route section",
        "execution": "not_executed",
        "gap": "catalog 已提供作者原名 Action；静态回放仍使用论文公开的精确 IRC route。",
    },
    {
        "step": 6,
        "paper_tool": "xyz_to_gjf",
        "candidate_actions": ["xyz_to_gjf"],
        "input": "优化 TS XYZ 和 IRC route",
        "output": "IRC GJF",
        "execution": "not_executed",
        "gap": "catalog 已提供作者原名 Action；本回放继续要求实际 Step 4 checkpoint 作为下游几何来源。",
    },
    {
        "step": 7,
        "paper_tool": "Gaussian",
        "candidate_actions": ["trace_intrinsic_reaction_coordinate"],
        "input": "优化 TS；B3LYP/6-31G(d), irc=(calcfc,LQA,maxpoints=50,recalc=5), IEFPCM acetone",
        "output": "连接正确反应物/产物的 IRC 轨迹",
        "execution": "not_executed",
        "gap": "精确 Gaussian IRC 需通过 native deck；本验证会等待本次 Step 4 的终态 checkpoint，但仍不能声称从论文未公开的 Step 1 初猜独立复现或已验证连接两端。",
    },
    {
        "step": 8,
        "paper_tool": "generate_gaussian_code",
        "candidate_actions": ["generate_gaussian_code"],
        "input": "高层 single-point energy calculation",
        "output": "高层 SP route section",
        "execution": "not_executed",
        "gap": "catalog 已提供作者原名 Action；静态回放仍使用论文公开的精确高层单点 route。",
    },
    {
        "step": 9,
        "paper_tool": "xyz_to_gjf",
        "candidate_actions": ["xyz_to_gjf"],
        "input": "优化 TS XYZ 和 SP route",
        "output": "SP GJF",
        "execution": "not_executed",
        "gap": "catalog 已提供作者原名 Action；本回放继续要求实际 Step 4 checkpoint 作为下游几何来源。",
    },
    {
        "step": 10,
        "paper_tool": "Gaussian",
        "candidate_actions": ["calculate_energy"],
        "input": "优化 TS；M06-2X/def2-TZVPP, SP, IEFPCM acetone, int=ultrafine, scf=tight",
        "output": "两个对映 TS 的高层电子能",
        "execution": "not_executed",
        "gap": "native SP 回放会 staged 本次 Step 4 的终态 checkpoint；它仍不等价于从论文未公开的 Step 1 初猜复现完整原始流水线。",
    },
    {
        "step": 11,
        "paper_tool": "Python script",
        "candidate_actions": ["derive_thermochemistry", "analyze_thermochemical_selectivity"],
        "input": "Step 4 热化学数据、Step 10 电子能、303.15 K",
        "output": "两条路径 ΔG‡ 和 ee%",
        "execution": "not_executed",
        "gap": "case1.json 是已解析结果，不是可供 GoodVibes/cclib 的 Gaussian 原始输出，且无 IRC/路径标签。",
    },
]


CASE1_NATIVE_REPLAY_PLAN = (
    {
        "stage": "step4_opt_freq",
        "paper_steps": (2, 3, 4),
        "route": "#p B3LYP/6-31G(d) Opt=(TS,NoEigenTest,CalcAll,MaxStep=5) Freq SCRF=(IEFPCM,Solvent=Acetone) Temperature=303.15",
        "calculation_intent": "transition_state",
        "cpu_cores": CASE1_NATIVE_CPU_CORES,
        "memory_mb": CASE1_NATIVE_MEMORY_MB,
        "gpu_count": CASE1_NATIVE_GPU_COUNT,
        "depends_on": None,
        "description": "论文 Step 4 的 B3LYP TS 优化与频率；Step 2/3 的 route/deck 由同一确定性 deck builder 生成。",
    },
    {
        "stage": "step7_irc",
        "paper_steps": (5, 6, 7),
        "route": "#p B3LYP/6-31G(d) IRC=(CalcFC,LQA,MaxPoints=50,ReCalc=5) SCRF=(IEFPCM,Solvent=Acetone)",
        "calculation_intent": "reaction_path",
        "cpu_cores": CASE1_NATIVE_CPU_CORES,
        "memory_mb": CASE1_NATIVE_MEMORY_MB,
        "gpu_count": CASE1_NATIVE_GPU_COUNT,
        "depends_on": "step4_opt_freq",
        "description": "论文 Step 7 的 B3LYP IRC；Step 5/6 的 route/deck 由同一确定性 deck builder 生成。",
    },
    {
        "stage": "step10_single_point",
        "paper_steps": (8, 9, 10),
        "route": "#p M062X/Def2TZVPP SP SCRF=(IEFPCM,Solvent=Acetone) Int=UltraFine SCF=Tight",
        "calculation_intent": "single_point",
        "cpu_cores": CASE1_NATIVE_CPU_CORES,
        "memory_mb": CASE1_NATIVE_MEMORY_MB,
        "gpu_count": CASE1_NATIVE_GPU_COUNT,
        "depends_on": "step4_opt_freq",
        "description": "论文 Step 10 的 M06-2X/def2-TZVPP 单点能；Step 8/9 的 route/deck 由同一确定性 deck builder 生成。",
    },
)


def load_case1_summary() -> dict[str, Any]:
    data = json.loads(CASE1_PATH.read_text(encoding="utf-8"))
    summary: dict[str, Any] = {"path": str(CASE1_PATH), "sections": {}}
    for section, records in data.items():
        summary["sections"][section] = {
            "record_count": len(records),
            "records": {},
        }
        for name, record in records.items():
            metadata = record.get("metadata", {})
            summary["sections"][section]["records"][name] = {
                "package": metadata.get("package"),
                "package_version": metadata.get("package_version"),
                "functional": metadata.get("functional"),
                "keywords": metadata.get("keywords", []),
                "charge": record.get("charge"),
                "multiplicity": record.get("mult"),
                "natom": record.get("natom"),
                "scfenergy_final": record.get("scfenergy_final"),
                "temperature": record.get("temperature"),
                "freeenergy": record.get("freeenergy"),
                "optdone": record.get("optdone"),
                "imaginary_frequency_count": sum(
                    1 for value in (record.get("vibfreqs") or []) if value < 0
                ),
                "has_coordinates": bool(record.get("atomcoords_final")),
            }
    return summary


def calculate_case1_delta_g_from_published_values(
    data: dict[str, Any], labels: tuple[str, str] = ("TS-2a", "TS-2b-1")
) -> dict[str, Any]:
    """按 case1.json 的真实单位回放两条路径的校正自由能和 ee。"""

    ev_to_hartree = 1.0 / 27.211386245988
    hartree_to_kcal = 627.509474
    corrected: dict[str, float] = {}
    for label in labels:
        opt = data["opt"][label]
        sp = data["sp"][label]
        corrected[label] = float(opt["freeenergy"]) + (
            float(sp["scfenergy_final"]) - float(opt["scfenergy_final"])
        ) * ev_to_hartree
    ordered = sorted(corrected.items(), key=lambda item: item[1])
    delta = (ordered[1][1] - ordered[0][1]) * hartree_to_kcal
    temperature = 303.15
    gas_constant_kcal = 0.001987204258
    ratio = math.exp(-abs(delta) / (gas_constant_kcal * temperature))
    ee = (1.0 - ratio) / (1.0 + ratio) * 100.0
    return {
        "corrected_free_energies_hartree": corrected,
        "lowest_two_labels": [item[0] for item in ordered],
        "delta_delta_g_kcal_mol": delta,
        "predicted_ee_percent": ee,
        "units": {
            "freeenergy": "Hartree",
            "scfenergy_final": "eV",
            "corrected_free_energies": "Hartree",
            "delta_delta_g": "kcal/mol",
        },
    }


def build_action_request(action_id: str, backend_id: str, inputs: dict[str, Any], *, settings: dict[str, Any] | None = None, method: dict[str, Any] | None = None) -> dict[str, Any]:
    """构造显式 Action 请求；不填写科学默认值。"""

    return {
        "action_id": action_id,
        "backend_id": backend_id,
        "component_backends": {},
        "source_id": None,
        "inputs": inputs,
        "method_spec": method or {},
        "action_settings": settings or {},
        "resource_limits": {"cpu_cores": 1, "memory_mb": 2048, "gpu_count": 0},
    }


def calculate_action_request_fingerprint(request: dict[str, Any]) -> str:
    """为同步 Action 请求计算稳定指纹，便于在没有 operation_id 的接口中定位结果。"""

    encoded = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def summarize_mcp_result(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    result = {
        "status": value.get("status"),
        "minichem_status": value.get("minichem_status"),
        "job_id": value.get("job_id"),
        "job_status": value.get("job_status"),
        "ready": value.get("ready"),
        "operation_id": value.get("operation_id"),
        "tool": value.get("tool"),
        "request_fingerprint": value.get("request_fingerprint"),
        "operation_fingerprint": value.get("operation_fingerprint"),
        "result": value.get("result"),
        "error": value.get("error"),
        "warnings": value.get("warnings"),
        "output_artifacts": value.get("output_artifacts"),
        "outputs": value.get("outputs"),
        "declared_artifacts": value.get("declared_artifacts"),
        "artifact_manifest": value.get("artifact_manifest"),
        "collection_manifest": value.get("collection_manifest"),
        "software_status": value.get("software_status"),
        "convergence_status": value.get("convergence_status"),
        "scientific_validation_status": value.get("scientific_validation_status"),
        "provenance": value.get("provenance"),
    }
    # 结果中可能有完整坐标矩阵；manifest 只保留小型摘要，ArtifactRef 仍完整保存。
    if isinstance(result.get("result"), dict):
        payload = result["result"]
        result["result"] = {
            key: payload[key]
            for key in (
                "energy_hartree",
                "converged",
                "atom_count",
                "conformer_count",
                "candidate_count",
                "imaginary_frequency_count",
                "free_energy_hartree",
                "atomic_action_trace",
                "transition_state_validated",
                "fully_grown",
                "path_files",
            )
            if key in payload
        }
    return result


def convert_case1_record_to_atomic_structure(record: dict[str, Any]) -> dict[str, Any]:
    """把附录公开的最终坐标转换为 MCP AtomicStructure 回放输入。"""

    atomnos = record.get("atomnos") or []
    coordinates = record.get("atomcoords_final") or []
    if len(atomnos) != len(coordinates):
        raise ValueError("case1.json atomnos 和 atomcoords_final 长度不一致")
    unknown = sorted({int(number) for number in atomnos if int(number) not in ATOMIC_SYMBOLS})
    if unknown:
        raise ValueError(f"case1.json 含脚本未声明的原子序数：{unknown}")
    return {
        "atoms": [
            {
                "element": ATOMIC_SYMBOLS[int(number)],
                "position_angstrom": [float(item) for item in row],
            }
            for number, row in zip(atomnos, coordinates)
        ],
        "charge": int(record.get("charge", 0)),
        "multiplicity": int(record.get("mult", 1)),
        "pbc": [False, False, False],
    }


def build_case1_gaussian_deck(
    record: dict[str, Any],
    *,
    route: str,
    title: str,
    checkpoint_name: str,
    parent_checkpoint_name: str | None = None,
    include_coordinates: bool = True,
    cpu_cores: int = CASE1_NATIVE_CPU_CORES,
    memory_mb: int = CASE1_NATIVE_MEMORY_MB,
) -> str:
    """生成确定性的 Gaussian deck；下游作业可从上游 checkpoint 读取几何。

    ``include_coordinates=True`` 时使用附录公开的 final coordinates。下游 IRC/SP
    必须传入 ``parent_checkpoint_name`` 并使用 ``Geom=AllCheck``，从 Step 4 的
    实际 checkpoint 读取几何，避免把公开坐标再次当作上游计算结果。
    """

    # 这是 Case 1 专用 builder；旧调用方即使传入 8/16 核，也不能生成
    # 不符合当前串行资源协议的 deck。
    del cpu_cores, memory_mb
    cpu_cores = CASE1_NATIVE_CPU_CORES
    memory_mb = CASE1_NATIVE_MEMORY_MB
    atomnos = record.get("atomnos") or []
    coordinates = record.get("atomcoords_final") or []
    if include_coordinates and (len(atomnos) != len(coordinates) or not atomnos):
        raise ValueError("Gaussian deck 需要长度一致且非空的 atomnos/atomcoords_final")
    if not include_coordinates and not parent_checkpoint_name:
        raise ValueError("从上游 checkpoint 生成 Gaussian deck 时必须提供 parent_checkpoint_name")
    effective_route = str(route)
    if not include_coordinates and "geom=allcheck" not in effective_route.casefold():
        effective_route = f"{effective_route} Geom=AllCheck"
    if not include_coordinates and "guess=read" not in effective_route.casefold():
        effective_route = f"{effective_route} Guess=Read"
    lines = [
        f"%NProcShared={int(cpu_cores)}",
        f"%Mem={int(memory_mb)}MB",
    ]
    if parent_checkpoint_name:
        lines.append(f"%OldChk={parent_checkpoint_name}")
    lines.append(f"%Chk={checkpoint_name}")
    lines.extend(
        [
            effective_route,
            "",
        ]
    )
    if not include_coordinates:
        # Geom=AllCheck 允许 Gaussian 从旧 checkpoint 读取标题、电荷、多重度和坐标。
        return "\n".join(lines) + "\n"
    charge = int(record.get("charge", 0))
    multiplicity = int(record.get("mult", 1))
    lines.extend(
        [
            title,
            "",
            f"{charge} {multiplicity}",
        ]
    )
    for number, position in zip(atomnos, coordinates):
        symbol = ATOMIC_SYMBOLS.get(int(number))
        if symbol is None or len(position) != 3:
            raise ValueError(f"无法为原子 {number!r} 生成三维 Gaussian 坐标")
        lines.append(
            f"{symbol:<2} {float(position[0]): .8f} {float(position[1]): .8f} {float(position[2]): .8f}"
        )
    # Gaussian 和 MiniChem lint 都要求坐标块和文件末尾有空行。
    return "\n".join(lines) + "\n\n"


def build_case1_native_request(
    *,
    deck_path: str,
    label: str,
    calculation_intent: str,
    parent_job_id: str | None = None,
    checkpoint_source_path: str | None = None,
    checkpoint_target_name: str | None = None,
    executable: str = "g16",
    cpu_cores: int = CASE1_NATIVE_CPU_CORES,
    memory_mb: int = CASE1_NATIVE_MEMORY_MB,
) -> dict[str, Any]:
    """构造 open_execution native 请求；operation_id 由适配器注入。

    ``checkpoint_source_path`` 用于把上游 native job 收集到的实际 checkpoint
    staged 到当前作业；请求中的 ``parent_job_id`` 保存同一条计算链的来源。
    """

    # native Gaussian 请求统一占满节点；参数保留用于兼容旧调用，但不允许
    # 旧的 8/16 核配置进入新提交。
    del cpu_cores, memory_mb
    cpu_cores = CASE1_NATIVE_CPU_CORES
    memory_mb = CASE1_NATIVE_MEMORY_MB
    staged_inputs = [{"source_path": deck_path, "target_path": "input.gjf"}]
    if checkpoint_source_path is not None:
        if not checkpoint_target_name:
            raise ValueError("checkpoint_source_path 存在时必须提供 checkpoint_target_name")
        staged_inputs.append(
            {
                "source_path": checkpoint_source_path,
                "target_path": checkpoint_target_name,
            }
        )

    return {
        "software_id": "gaussian",
        "executable": executable,
        "arguments": [],
        "staged_inputs": staged_inputs,
        "stdin_target": "input.gjf",
        "resource_limits": {
            "cpu_cores": int(cpu_cores),
            "memory_mb": int(memory_mb),
            "gpu_count": CASE1_NATIVE_GPU_COUNT,
        },
        "calculation_intent": calculation_intent,
        "label": label,
        "parent_job_id": parent_job_id,
    }


def build_native_operation_id(label: str, stage: str, *, attempt_index: int = 1) -> str:
    """返回可持久化的 Case 1 native operation identity。"""

    normalized = label.lower().replace("-", "_")
    base = f"case1-{normalized}-{stage.replace('_', '-')}"
    return base if attempt_index == 1 else f"{base}-attempt-{attempt_index}"


def case1_native_resource_limits(plan: dict[str, Any] | None = None) -> dict[str, int]:
    """返回 Case 1 每个 Gaussian 作业统一占满的资源限制。

    ``plan`` 仅用于兼容旧 manifest；即使旧计划记录了 8/16 核，也不能把
    旧配置带入新的提交。返回值会被同时写入 Gaussian deck、native 请求和
    运行记录，便于恢复时核对实际申请的资源。
    """

    del plan
    return {
        "cpu_cores": CASE1_NATIVE_CPU_CORES,
        "memory_mb": CASE1_NATIVE_MEMORY_MB,
        "gpu_count": CASE1_NATIVE_GPU_COUNT,
    }


def _native_record_is_active(record: Any) -> bool:
    """判断 native 记录是否仍占用串行槽位或尚未获服务端终态确认。"""

    if not isinstance(record, dict) or record.get("archived_for_retry"):
        return False
    status = str(record.get("status", "")).casefold()
    # 取消、失败等文字状态只有在服务端明确确认 terminal/ready 后才释放
    # 串行槽位；显式的未确认标记优先于其他摘要字段。
    if record.get("terminal") is False or record.get("ready") is False:
        return True
    server_job_status = str(record.get("job_status", "")).casefold()
    if server_job_status in _NATIVE_ACTIVE_STATUSES:
        return True
    if status in _NATIVE_ACTIVE_STATUSES:
        return True
    if status == "completed":
        job_status = str(record.get("job_status", "")).casefold()
        return bool(
            record.get("ready") is False
            or job_status in _NATIVE_ACTIVE_STATUSES
            or record.get("terminal") is False
        )
    if status in {"uncertain_requires_manual_recovery", "recovery_failed"}:
        return bool(record.get("operation_id") or record.get("submission_attempted"))
    if status in {"call_failed", "technical_pause"}:
        outcome = str(
            record.get("minichem_status")
            or record.get("job_status")
            or record.get("terminal_status")
            or ""
        ).casefold()
        # 只有服务端明确给出失败、超时或取消，才释放串行槽位；其余
        # technical_pause/call_failed 都可能是在提交后断联，必须继续阻塞。
        if outcome in _NATIVE_TERMINAL_FAILURE_STATUSES:
            return False
        return bool(record.get("operation_id") or record.get("submission_attempted"))
    if status in _NATIVE_TERMINAL_FAILURE_STATUSES:
        return False
    if status in {
        "native_validation_failed",
        "runtime_unavailable",
        "blocked_dependency",
        "serial_blocked",
        "pending",
        "not_executed",
    }:
        return False
    # 带 operation_id 或已写入提交意图的未知状态不能被乐观地视为终态。
    return bool(record.get("operation_id") or record.get("submission_attempted"))


def find_active_case1_native_operation(
    runs: dict[str, Any], *, exclude_key: str | None = None
) -> tuple[str, dict[str, Any]] | None:
    """按 manifest 顺序返回当前唯一仍活动的 Gaussian operation。"""

    for key, value in runs.items():
        if key == exclude_key or not key.startswith("native_"):
            continue
        if not isinstance(value, dict) or value.get("archived_for_retry"):
            continue
        if _native_record_is_active(value):
            return key, value
    return None


def summarize_native_validation(value: Any) -> dict[str, Any]:
    """保留 native preflight 的可审计字段，不把本机绝对路径写入 manifest。"""

    if not isinstance(value, dict):
        return {"status": "invalid_response", "value_type": type(value).__name__}
    result: dict[str, Any] = {
        key: value.get(key)
        for key in ("status", "valid", "automatic_fallback", "error")
        if key in value
    }
    for key in ("lint", "validation", "request_fingerprint", "operation_id"):
        if key in value:
            result[key] = value[key]
    return result


def check_native_job_terminal(value: Any) -> bool:
    """判断 native 结果是否已完成并可作为下游作业输入。"""

    if not isinstance(value, dict) or value.get("status") != "completed":
        return False
    return value.get("ready") is not False


def check_native_job_scientific_gate(value: Any) -> bool:
    """判断 native 作业是否满足对应计算意图的机械科学判据。"""

    return bool(
        check_native_job_terminal(value)
        and isinstance(value, dict)
        and value.get("software_status") == "normal_termination"
        and value.get("convergence_status") == "converged"
        and value.get("scientific_validation_status") == "mechanically_valid"
    )


def _iter_native_result_mappings(value: Any) -> list[dict[str, Any]]:
    """遍历 native 收据中的有限嵌套对象，读取可选的路径验证字段。"""

    if not isinstance(value, dict):
        return []
    mappings = [value]
    for key in (
        "result",
        "execution_status_details",
        "details",
        "convergence",
        "irc_connectivity",
        "endpoint_connectivity",
        "connectivity",
    ):
        nested = value.get(key)
        if isinstance(nested, dict):
            mappings.extend(_iter_native_result_mappings(nested))
    return mappings


def assess_case1_irc_connectivity_evidence(value: Any) -> str:
    """评估 IRC 收据是否明确报告了双向路径连通性。

    返回 ``validated``、``failed`` 或 ``not_reported``。MiniChem 当前的
    Gaussian mechanical gate 只保证 IRC 计算正常结束；如果服务额外返回
    端点或正反向字段，脚本会把这些字段作为独立连通性证据核对。缺少这些
    可选字段时不把结果伪装成失败，但完整路径记录会明确标记证据未报告。
    """

    direct_keys = (
        "irc_bidirectional_completed",
        "bidirectional_irc_completed",
        "irc_connected",
        "endpoints_connected",
        "endpoint_connectivity_validated",
        "transition_state_connectivity_validated",
        "connectivity_validated",
        "irc_connectivity_status",
        "path_connectivity_status",
        "endpoint_connectivity_status",
    )
    paired_keys = (
        ("forward_completed", "reverse_completed"),
        ("forward_path_complete", "reverse_path_complete"),
        ("irc_forward_completed", "irc_reverse_completed"),
        ("forward_connected", "reverse_connected"),
        ("reactant_endpoint_connected", "product_endpoint_connected"),
        ("forward", "reverse"),
        ("reactant", "product"),
    )
    true_values = {"true", "yes", "passed", "pass", "completed", "validated", "connected"}
    false_values = {"false", "no", "failed", "fail", "incomplete", "invalid", "disconnected"}
    saw_explicit = False
    saw_success = False
    saw_failure = False
    for mapping in _iter_native_result_mappings(value):
        for key in direct_keys:
            raw = mapping.get(key)
            if isinstance(raw, bool):
                saw_explicit = True
                if raw:
                    saw_success = True
                else:
                    saw_failure = True
            elif isinstance(raw, str):
                normalized = raw.strip().casefold()
                if normalized in true_values:
                    saw_explicit = True
                    saw_success = True
                if normalized in false_values:
                    saw_explicit = True
                    saw_failure = True
        for first_key, second_key in paired_keys:
            first = mapping.get(first_key)
            second = mapping.get(second_key)
            first_normalized = first.strip().casefold() if isinstance(first, str) else first
            second_normalized = second.strip().casefold() if isinstance(second, str) else second
            first_pass = first is True or first_normalized in true_values
            second_pass = second is True or second_normalized in true_values
            first_fail = first is False or first_normalized in false_values
            second_fail = second is False or second_normalized in false_values
            if (first_pass or first_fail) and (second_pass or second_fail):
                saw_explicit = True
                if first_pass and second_pass:
                    saw_success = True
                else:
                    saw_failure = True
    if saw_failure or saw_explicit:
        return "failed" if saw_failure else "validated"
    if saw_success:
        return "validated"
    return "not_reported"


def assess_case1_native_dependency_link(
    parent_result: dict[str, Any] | None,
    child_result: dict[str, Any] | None,
) -> str:
    """核对下游 native 收据是否声明并复用了上游 job/checkpoint。

    返回 ``validated``、``mismatch``、``missing_parent_output`` 或
    ``not_recorded``。旧 manifest 可能没有链路摘要，因此缺字段时保留
    ``not_recorded``，由调用方单独展示风险而不把 Step 11 数值分析拦住。
    """

    if not isinstance(parent_result, dict) or not isinstance(child_result, dict):
        return "not_recorded"
    expected_operation = parent_result.get("operation_id")
    expected_job = parent_result.get("job_id")
    expected_checkpoint = parent_result.get("checkpoint_path") or extract_native_checkpoint_path(
        parent_result
    )
    if expected_operation is None and expected_job is None and expected_checkpoint is None:
        # 旧 manifest 只保存状态和日志摘要时无法回溯链路；这与已知链路
        # 缺少 checkpoint 不同，不能凭空把旧记录判成错误。
        return "not_recorded"
    if not expected_operation or not expected_job or not expected_checkpoint:
        return "missing_parent_output"
    actual_operation = child_result.get("parent_operation_id")
    actual_job = child_result.get("parent_job_id")
    actual_checkpoint = child_result.get("parent_checkpoint_path")
    if actual_operation is None and actual_job is None and actual_checkpoint is None:
        return "not_recorded"
    if (
        actual_operation != expected_operation
        or actual_job != expected_job
        or actual_checkpoint != expected_checkpoint
    ):
        return "mismatch"
    return "validated"


def _collect_native_output_items(value: Any) -> list[dict[str, Any]]:
    """提取 native collect 返回的文件条目，避免把收据原文写入下游请求。"""

    if not isinstance(value, dict):
        return []
    items: list[dict[str, Any]] = []
    for key in ("outputs", "declared_artifacts", "output_artifacts"):
        candidate = value.get(key)
        if isinstance(candidate, list):
            items.extend(item for item in candidate if isinstance(item, dict))
    artifact_manifest = value.get("artifact_manifest")
    if isinstance(artifact_manifest, dict):
        artifacts = artifact_manifest.get("artifacts")
        if isinstance(artifacts, list):
            items.extend(item for item in artifacts if isinstance(item, dict))
    nested = value.get("result")
    if isinstance(nested, dict):
        items.extend(_collect_native_output_items(nested))
    return items


def extract_native_output_paths(value: Any, suffix: str | None = None) -> list[str]:
    """按收集顺序返回 native 作业产生的 workspace 相对路径。"""

    paths: list[str] = []
    seen: set[str] = set()
    if isinstance(value, dict):
        # 运行记录会把常用日志/checkpoint 路径提升到顶层；恢复旧 manifest
        # 时可能只剩这些摘要字段，不能因为缺少 outputs 就丢失可交接文件。
        for name in ("log_path", "stdout_path", "checkpoint_path"):
            path = value.get(name)
            if (
                isinstance(path, str)
                and path
                and (suffix is None or path.casefold().endswith(suffix.casefold()))
            ):
                paths.append(path)
                seen.add(path)
    for item in _collect_native_output_items(value):
        path = next(
            (
                item.get(name)
                for name in ("path", "workspace_path", "source_path", "job_relative_path")
                if isinstance(item.get(name), str) and item.get(name)
            ),
            None,
        )
        if not isinstance(path, str) or (suffix and not path.casefold().endswith(suffix.casefold())):
            continue
        if path not in seen:
            paths.append(path)
            seen.add(path)
    return paths


def extract_native_checkpoint_path(value: Any) -> str | None:
    """返回 native 收集结果中的 checkpoint 路径，缺失时返回 None。"""

    paths = extract_native_output_paths(value, suffix=".chk")
    return paths[0] if paths else None


def extract_native_log_path(value: Any) -> str | None:
    """返回 native 收集结果中的日志路径，优先使用 stdout.log。"""

    paths = extract_native_output_paths(value, suffix=".log")
    for path in paths:
        if Path(path).name.casefold() == "stdout.log":
            return path
    return paths[0] if paths else None


def _is_sha256_text(value: Any) -> bool:
    """判断输出记录是否带有可核对的 SHA-256 文本。"""

    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdefABCDEF" for character in value)


def _find_native_output_evidence(value: Any, path: str) -> dict[str, Any] | None:
    """在 native 收据中找到指定输出路径及其大小、哈希证据。"""

    for item in _collect_native_output_items(value):
        candidate = next(
            (
                item.get(name)
                for name in ("path", "workspace_path", "source_path", "job_relative_path")
                if isinstance(item.get(name), str) and item.get(name)
            ),
            None,
        )
        if candidate != path:
            continue
        if not _is_sha256_text(item.get("sha256")):
            return None
        size_bytes = item.get("size_bytes")
        if type(size_bytes) is not int or size_bytes < 0:
            return None
        return item
    return None


def _resolve_operation_receipt_path(
    receipt_path: str, operation_id: str, operation_root: Path | None
) -> Path | None:
    """解析并确认 operation 收据路径确实存在且不是符号链接。"""

    expected_name = f"{operation_id}.json"
    candidate = Path(receipt_path)
    if candidate.name != expected_name:
        return None
    candidates = [candidate]
    if operation_root is not None and not candidate.is_absolute():
        candidates.extend(
            [operation_root / candidate, operation_root / expected_name]
        )
    for item in candidates:
        if item.is_file() and not item.is_symlink():
            return item
    return None


def validate_case1_native_stage_identity(
    value: Any,
    *,
    operation_root: Path | None = None,
) -> dict[str, Any]:
    """严格核对已完成 native stage 的 operation、job、checkpoint 和日志身份。

    返回 ``valid``、身份字段和结构化错误。即使调用方没有提供本地
    ``operation_root``，也要求收据文件名、输出路径、大小和 SHA-256 能够在
    manifest 中相互核对；提供根目录时还会读取收据并核对其中的 operation/job。
    """

    errors: list[str] = []
    result = value if isinstance(value, dict) else {}
    operation_id = result.get("operation_id")
    job_id = result.get("job_id")
    if not isinstance(operation_id, str) or not operation_id.strip():
        errors.append("missing_operation_id")
    if not isinstance(job_id, str) or not job_id.strip():
        errors.append("missing_job_id")

    receipt_path = result.get("operation_receipt_path")
    receipt_file: Path | None = None
    if not isinstance(receipt_path, str) or not receipt_path.strip():
        errors.append("missing_operation_receipt_path")
    elif isinstance(operation_id, str) and operation_id.strip():
        receipt_file = _resolve_operation_receipt_path(
            receipt_path, operation_id, operation_root
        )
        if receipt_file is None:
            errors.append("operation_receipt_unverifiable")

    checkpoint_path = extract_native_checkpoint_path(result)
    log_path = extract_native_log_path(result)
    if not checkpoint_path:
        errors.append("missing_checkpoint_path")
    if not log_path:
        errors.append("missing_log_path")
    checkpoint_evidence = (
        _find_native_output_evidence(result, checkpoint_path)
        if checkpoint_path
        else None
    )
    log_evidence = (
        _find_native_output_evidence(result, log_path) if log_path else None
    )
    if checkpoint_path and checkpoint_evidence is None:
        errors.append("checkpoint_evidence_unverifiable")
    if log_path and log_evidence is None:
        errors.append("log_evidence_unverifiable")

    receipt_payload: dict[str, Any] | None = None
    if receipt_file is not None:
        try:
            loaded = json.loads(receipt_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            errors.append("operation_receipt_unreadable")
        else:
            if isinstance(loaded, dict):
                receipt_payload = loaded
                if loaded.get("operation_id") != operation_id:
                    errors.append("operation_receipt_operation_mismatch")
                if loaded.get("job_id") not in {None, job_id}:
                    errors.append("operation_receipt_job_mismatch")
            else:
                errors.append("operation_receipt_not_object")

    return {
        "valid": not errors,
        "errors": errors,
        "operation_id": operation_id,
        "job_id": job_id,
        "operation_receipt_path": receipt_path,
        "checkpoint_path": checkpoint_path,
        "log_path": log_path,
        "checkpoint_evidence": checkpoint_evidence,
        "log_evidence": log_evidence,
        "operation_receipt": receipt_payload,
    }


def _build_native_record_metadata(
    *,
    operation_id: str,
    request_fingerprint: str,
    deck_path: str,
    plan: dict[str, Any],
    operation_root: Path,
    attempt_index: int = 1,
    parent_operation_id: str | None = None,
    parent_job_id: str | None = None,
    parent_checkpoint_path: str | None = None,
    parent_checkpoint_target: str | None = None,
) -> dict[str, Any]:
    """构造 native 运行记录的稳定元数据。"""

    resource_limits = case1_native_resource_limits(plan)
    return {
        "operation_id": operation_id,
        "request_fingerprint": request_fingerprint,
        "deck_path": deck_path,
        "paper_steps": list(plan["paper_steps"]),
        "route": plan["route"],
        "calculation_intent": plan["calculation_intent"],
        "resource_limits": resource_limits,
        "operation_receipt_path": str(operation_root / f"{operation_id}.json"),
        "attempt_index": attempt_index,
        "depends_on": plan.get("depends_on"),
        "parent_operation_id": parent_operation_id,
        "parent_job_id": parent_job_id,
        "parent_checkpoint_path": parent_checkpoint_path,
        "parent_checkpoint_target": parent_checkpoint_target,
        "input_chaining": bool(parent_checkpoint_path),
        "reference_coordinate_source": "supplementary/case1.json final coordinates (仅 Step 4 初始回放)",
        "scientific_scope": "Step 4 使用公开 final coordinates；Step 7/10 只使用本次 Step 4 收集的 checkpoint。",
    }


def _enrich_native_result(
    value: Any,
    *,
    metadata: dict[str, Any],
    recovered_without_resubmission: bool = False,
) -> dict[str, Any]:
    """把可复验的输出路径和终态标记合并进 native manifest 记录。"""

    result = summarize_mcp_result(value)
    result.update(metadata)
    if recovered_without_resubmission:
        result["recovered_without_resubmission"] = True
    checkpoint_path = extract_native_checkpoint_path(result)
    log_path = extract_native_log_path(result)
    if checkpoint_path:
        result["checkpoint_path"] = checkpoint_path
    if log_path:
        result["log_path"] = log_path
    result["terminal"] = check_native_job_terminal(result)
    result["scientifically_valid"] = check_native_job_scientific_gate(result)
    result["terminal_output_paths"] = extract_native_output_paths(result)
    return result


def _native_failure_can_be_retried(record: Any) -> bool:
    """判断 native 记录是否是已确认终态失败且允许显式新尝试。"""

    if not isinstance(record, dict) or record.get("archived_for_retry"):
        return False
    status = str(record.get("status", "")).casefold()
    outcome = str(
        record.get("minichem_status")
        or record.get("job_status")
        or record.get("terminal_status")
        or ""
    ).casefold()
    if status in {"technical_pause", "failed", "timeout", "timed_out", "cancelled", "canceled"}:
        return outcome in _NATIVE_TERMINAL_FAILURE_STATUSES or status in _NATIVE_TERMINAL_FAILURE_STATUSES
    return bool(
        status in {"native_validation_failed", "runtime_unavailable"}
        and not record.get("submission_attempted")
    )


def _record_case1_serial_block(
    *,
    runs: dict[str, Any],
    key: str,
    existing: dict[str, Any] | None,
    operation_id: str,
    attempt_index: int,
    deck_path: str,
    plan: dict[str, Any],
    active_key: str,
    active_record: dict[str, Any],
    save_manifest: Any,
) -> dict[str, Any]:
    """记录等待前一 Gaussian operation 终态的可恢复串行阻塞。"""

    blocked = dict(existing) if isinstance(existing, dict) else {}
    if "serial_blocked_previous_status" not in blocked:
        blocked["serial_blocked_previous_status"] = (
            existing.get("status") if isinstance(existing, dict) else None
        )
    if "serial_blocked_previous_submission_attempted" not in blocked:
        blocked["serial_blocked_previous_submission_attempted"] = bool(
            existing.get("submission_attempted") if isinstance(existing, dict) else False
        )
    blocked.update(
        {
            "status": "serial_blocked",
            "serial_blocked": True,
            "serial_blocked_by_key": active_key,
            "serial_blocked_by_operation_id": active_record.get("operation_id"),
            "serial_blocked_reason": (
                "前一 Gaussian operation 尚未由服务端确认终态；本次仅记录串行阻塞，"
                "未写入 deck、未调用 validate/submit。"
            ),
            "operation_id": operation_id,
            "attempt_index": attempt_index,
            "deck_path": deck_path,
            "paper_steps": list(plan["paper_steps"]),
            "route": plan["route"],
            "calculation_intent": plan["calculation_intent"],
            "resource_limits": case1_native_resource_limits(plan),
            "depends_on": plan.get("depends_on"),
            "submission_attempted": False,
            "serial_blocked_at": time.time(),
        }
    )
    runs[key] = blocked
    save_manifest()
    return blocked


def run_case1_native_step(
    *,
    adapter: MiniChemMcpAdapter,
    runs: dict[str, Any],
    save_manifest: Any,
    label: str,
    record: dict[str, Any],
    plan: dict[str, Any],
    gaussian_available: bool,
    operation_root: Path,
    dependency_result: dict[str, Any] | None = None,
    retry_failed: bool = False,
) -> dict[str, Any]:
    """按依赖顺序预验证并提交一个 Gaussian native job。

    下游作业只有在 ``dependency_result`` 已收集到终态 checkpoint 后才会写入
    deck、调用 preflight 或 submit；已有 operation 只调用 status/collect。
    """

    stage = str(plan["stage"])
    key = f"native_{label}_{stage}"
    safe_label = label.lower().replace("-", "_")
    existing = runs.get(key)

    # 串行阻塞只保存等待事实；前一 operation 终态后，本次运行自动清除
    # 标记并继续同一 attempt，保持 operation_id 不变。
    if isinstance(existing, dict) and existing.get("status") == "serial_blocked":
        active = find_active_case1_native_operation(runs, exclude_key=key)
        if active is not None:
            return _record_case1_serial_block(
                runs=runs,
                key=key,
                existing=existing,
                operation_id=str(existing.get("operation_id"))
                if isinstance(existing.get("operation_id"), str)
                else build_native_operation_id(label, stage),
                attempt_index=int(existing.get("attempt_index", 1)),
                deck_path=str(existing.get("deck_path", "")),
                plan=plan,
                active_key=active[0],
                active_record=active[1],
                save_manifest=save_manifest,
            )
        previous_status = existing.get("serial_blocked_previous_status")
        previous_submission_attempted = existing.get(
            "serial_blocked_previous_submission_attempted", False
        )
        existing["status"] = previous_status or "pending"
        existing["submission_attempted"] = bool(previous_submission_attempted)
        existing.pop("serial_blocked", None)
        existing["serial_unblocked_at"] = time.time()

    attempt_index = (
        int(existing.get("attempt_index", 1))
        if isinstance(existing, dict)
        else 1
    )
    existing_status = existing.get("status") if isinstance(existing, dict) else None

    attempt_suffix = "" if attempt_index == 1 else f"_attempt_{attempt_index}"
    deck_path = f"code/case1_native/{safe_label}_{stage}{attempt_suffix}.gjf"
    checkpoint_name = f"{safe_label}_{stage}{attempt_suffix}.chk"
    operation_id = (
        str(existing.get("operation_id"))
        if isinstance(existing, dict)
        and isinstance(existing.get("operation_id"), str)
        else build_native_operation_id(label, stage, attempt_index=attempt_index)
    )

    retryable_terminal_failure = _native_failure_can_be_retried(existing)
    current_is_active = _native_record_is_active(existing)
    active_other = (
        None
        if current_is_active
        else find_active_case1_native_operation(runs, exclude_key=key)
    )
    wants_submission = bool(
        existing is None
        or existing_status in {"pending", "blocked_dependency"}
        or (retry_failed and retryable_terminal_failure)
    )
    if active_other is not None and wants_submission:
        # 先前明确取消/失败的 attempt 不在串行阻塞期间重试；保留其原始
        # job_id 和服务端状态，待解锁后的下一次运行再归档并生成新 attempt。
        return _record_case1_serial_block(
            runs=runs,
            key=key,
            existing=existing if isinstance(existing, dict) else None,
            operation_id=operation_id,
            attempt_index=attempt_index,
            deck_path=deck_path,
            plan=plan,
            active_key=active_other[0],
            active_record=active_other[1],
            save_manifest=save_manifest,
        )

    # 仅在没有前序活动 operation 时归档明确失败的 attempt 并开始新尝试。
    if retry_failed and retryable_terminal_failure:
        archive_key = f"{key}_attempt_{attempt_index}"
        if archive_key in runs:
            raise RuntimeError(f"native 失败尝试归档键已存在：{archive_key}")
        archived = dict(existing)
        archived["archived_for_retry"] = True
        archived["archived_at"] = time.time()
        runs[archive_key] = archived
        attempt_index += 1
        existing = None
        existing_status = None
        attempt_suffix = f"_attempt_{attempt_index}"
        deck_path = f"code/case1_native/{safe_label}_{stage}{attempt_suffix}.gjf"
        checkpoint_name = f"{safe_label}_{stage}{attempt_suffix}.chk"
        operation_id = build_native_operation_id(label, stage, attempt_index=attempt_index)
        save_manifest()

    # 已经提交过的 operation 优先恢复，避免因为依赖元数据暂时缺失而重新提交。
    existing_attempted = isinstance(existing, dict) and (
        existing.get("submission_attempted")
        or _native_record_is_active(existing)
        or existing_status == "completed"
    )
    if isinstance(existing, dict) and existing_attempted:
        metadata = _build_native_record_metadata(
            operation_id=operation_id,
            request_fingerprint=str(existing.get("request_fingerprint", "")),
            deck_path=str(existing.get("deck_path", deck_path)),
            plan=plan,
            operation_root=operation_root,
            attempt_index=attempt_index,
            parent_operation_id=existing.get("parent_operation_id"),
            parent_job_id=existing.get("parent_job_id"),
            parent_checkpoint_path=existing.get("parent_checkpoint_path"),
            parent_checkpoint_target=existing.get("parent_checkpoint_target"),
        )
        # 旧的 completed 记录可能确实使用过 8/16 核；恢复只读证据时保留
        # 原申请，新的提交路径仍由统一 Case 1 资源常量生成。
        if isinstance(existing.get("resource_limits"), dict):
            metadata["resource_limits"] = dict(existing["resource_limits"])
        status = existing.get("status")
        if status in {"waiting", "completed", "recovery_failed"} and existing.get("operation_id"):
            try:
                resumed = adapter.resume_operation(operation_id, collect=True)
                runs[key] = _enrich_native_result(
                    resumed,
                    metadata=metadata,
                    recovered_without_resubmission=True,
                )
            except Exception as exc:  # pragma: no cover - live service diagnostic
                existing.update(
                    {
                        "status": "recovery_failed",
                        "recovered_without_resubmission": True,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                runs[key] = existing
            save_manifest()
            return runs[key]
        if _native_record_is_active(existing):
            receipt_path = operation_root / f"{operation_id}.json"
            if receipt_path.is_file() and existing.get("operation_id"):
                try:
                    resumed = adapter.resume_operation(operation_id, collect=True)
                    runs[key] = _enrich_native_result(
                        resumed,
                        metadata=metadata,
                        recovered_without_resubmission=True,
                    )
                except Exception as exc:  # pragma: no cover - live service diagnostic
                    existing.update(
                        {
                            "status": "recovery_failed",
                            "recovered_without_resubmission": True,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    runs[key] = existing
            else:
                existing["status"] = "uncertain_requires_manual_recovery"
                existing["recovery_required"] = True
                existing.setdefault(
                    "recovery_note",
                    "已有 native 尝试但没有 operation receipt；脚本不会自动再次 validate/submit。",
                )
                runs[key] = existing
            save_manifest()
            return runs[key]
        save_manifest()
        return existing

    if isinstance(existing, dict) and existing_status not in {
        None,
        "blocked_dependency",
        "pending",
    }:
        # validation_failed、runtime_unavailable、call_failed 等失败证据需要人工
        # 改变计划后再重试；只有从属阻塞可以在上游完成后自动解锁。
        save_manifest()
        return existing

    parent_job_id: str | None = None
    parent_operation_id: str | None = None
    parent_checkpoint_path: str | None = None
    parent_checkpoint_target: str | None = None
    dependency_stage = plan.get("depends_on")
    if dependency_stage:
        if not check_native_job_terminal(dependency_result):
            blocked = dict(existing) if isinstance(existing, dict) else {}
            blocked.update(
                {
                    "status": "blocked_dependency",
                    "operation_id": operation_id,
                    "deck_path": deck_path,
                    "paper_steps": list(plan["paper_steps"]),
                    "route": plan["route"],
                    "calculation_intent": plan["calculation_intent"],
                    "depends_on": dependency_stage,
                    "submission_attempted": False,
                    "blocked_reason": "上游 native 作业尚未完成；未写入 deck、未调用 validate/submit。",
                }
            )
            blocked["resource_limits"] = case1_native_resource_limits(plan)
            runs[key] = blocked
            save_manifest()
            return blocked
        if not check_native_job_scientific_gate(dependency_result):
            blocked = dict(existing) if isinstance(existing, dict) else {}
            blocked.update(
                {
                    "status": "blocked_dependency",
                    "operation_id": operation_id,
                    "deck_path": deck_path,
                    "paper_steps": list(plan["paper_steps"]),
                    "route": plan["route"],
                    "calculation_intent": plan["calculation_intent"],
                    "depends_on": dependency_stage,
                    "submission_attempted": False,
                    "blocked_reason": (
                        "上游 native 作业已经结束，但未通过软件正常终止、收敛状态和"
                        "机械科学有效性检查；未写入 deck、未调用 validate/submit。"
                    ),
                }
            )
            blocked["resource_limits"] = case1_native_resource_limits(plan)
            runs[key] = blocked
            save_manifest()
            return blocked
        parent_job_id = dependency_result.get("job_id") if isinstance(dependency_result, dict) else None
        parent_operation_id = (
            dependency_result.get("operation_id")
            if isinstance(dependency_result, dict)
            else None
        )
        parent_checkpoint_path = extract_native_checkpoint_path(dependency_result)
        if not parent_operation_id or not parent_job_id or not parent_checkpoint_path:
            blocked = dict(existing) if isinstance(existing, dict) else {}
            blocked.update(
                {
                    "status": "blocked_dependency",
                    "operation_id": operation_id,
                    "deck_path": deck_path,
                    "paper_steps": list(plan["paper_steps"]),
                    "route": plan["route"],
                    "calculation_intent": plan["calculation_intent"],
                    "depends_on": dependency_stage,
                    "submission_attempted": False,
                    "blocked_reason": "上游已返回 completed，但缺少 operation_id、job_id 或可收集的 .chk checkpoint。",
                }
            )
            blocked["resource_limits"] = case1_native_resource_limits(plan)
            runs[key] = blocked
            save_manifest()
            return blocked
        parent_checkpoint_target = Path(parent_checkpoint_path).name

    deck = build_case1_gaussian_deck(
        record,
        route=str(plan["route"]),
        title=f"ARCHE Case 1 {label} {stage}",
        checkpoint_name=checkpoint_name,
        parent_checkpoint_name=parent_checkpoint_target,
        include_coordinates=not bool(parent_checkpoint_path),
        cpu_cores=CASE1_NATIVE_CPU_CORES,
        memory_mb=CASE1_NATIVE_MEMORY_MB,
    )
    request = build_case1_native_request(
        deck_path=deck_path,
        label=f"ARCHE Case 1 {label} {stage}",
        calculation_intent=str(plan["calculation_intent"]),
        parent_job_id=parent_job_id,
        checkpoint_source_path=parent_checkpoint_path,
        checkpoint_target_name=parent_checkpoint_target,
        cpu_cores=CASE1_NATIVE_CPU_CORES,
        memory_mb=CASE1_NATIVE_MEMORY_MB,
    )
    request_fingerprint = calculate_action_request_fingerprint(request)
    metadata = _build_native_record_metadata(
        operation_id=operation_id,
        request_fingerprint=request_fingerprint,
        deck_path=deck_path,
        plan=plan,
        operation_root=operation_root,
        attempt_index=attempt_index,
        parent_operation_id=parent_operation_id,
        parent_job_id=parent_job_id,
        parent_checkpoint_path=parent_checkpoint_path,
        parent_checkpoint_target=parent_checkpoint_target,
    )
    record = {
        "status": "preparing",
        **metadata,
        "submission_attempted": False,
        "reference_coordinate_source": "supplementary/case1.json final coordinates (仅 Step 4 初始回放)",
        "scientific_scope": "Step 4 使用公开 final coordinates；Step 7/10 只使用本次 Step 4 收集的 checkpoint。",
        "started_at": time.time(),
    }
    runs[key] = record
    save_manifest()

    if not gaussian_available:
        record["status"] = "runtime_unavailable"
        record["software_id"] = "gaussian"
        record["reason"] = "open_execution 清单未报告 Gaussian 可提交；未写入 deck、未调用 validate_native_job/submit_native_job。"
        record["checked_at"] = time.time()
        save_manifest()
        return record

    try:
        authored = adapter.ensure_workspace_text(deck_path, deck)
        record["deck_write"] = {
            item: authored.get(item)
            for item in ("status", "path", "size_bytes", "sha256", "reused")
            if item in authored
        }
        record["status"] = "validating"
        save_manifest()
        validation = adapter.validate_native_job(request, operation_id)
        record["native_validation"] = summarize_native_validation(validation)
        if validation.get("status") != "success" or validation.get("valid") is False:
            record["status"] = "native_validation_failed"
            save_manifest()
            return record
        # 先落盘提交意图，崩溃恢复时有明确证据，不会重新发起副作用。
        record["status"] = "submitting"
        record["submission_attempted"] = True
        save_manifest()
        submitted = adapter.submit_native_job(request, operation_id)
        runs[key] = _enrich_native_result(
            submitted,
            metadata={**metadata, "native_validation": summarize_native_validation(validation)},
        )
        runs[key]["submission_attempted"] = True
    except Exception as exc:  # pragma: no cover - live service diagnostic
        record.update(
            {
                "status": "call_failed",
                "submission_attempted": record.get("submission_attempted", False),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        runs[key] = record
    save_manifest()
    return runs[key]


def build_case1_fresh_analysis_program(input_names: list[str]) -> str:
    """生成只读取本次 Step 4/10 native 日志的 Step 11 分析程序。"""

    if not input_names:
        raise ValueError("fresh analysis 至少需要一个 native 日志输入")
    input_lines = "\n".join(
        f'    {name!r}: ctx.input({name!r}).read_text(encoding="utf-8"),'
        for name in input_names
    )
    return f'''from __future__ import annotations

import json
import math
import re
from researchchem_job import JobContext

ctx = JobContext.load()
raw_logs = {{
{input_lines}
}}

def parse_gaussian_log(text):
    number_pattern = r"[-+]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[DEde][-+]?\\d+)?"
    scf_values = re.findall(
        rf"SCF Done:\\s+E\\([^)]*\\)\\s+=\\s+({{number_pattern}})",
        text,
    )
    free_values = re.findall(
        rf"Sum of electronic and thermal Free Energies=\\s*({{number_pattern}})",
        text,
    )
    nimag_values = re.findall(r"NImag=\\s*(\\d+)", text)
    if not nimag_values:
        nimag_values = re.findall(r"NImag\\s+=\\s*(\\d+)", text)
    frequency_values = []
    for line in text.splitlines():
        if not re.search(r"^\\s*Frequencies\\s+--", line, flags=re.IGNORECASE):
            continue
        _, _, frequency_text = line.partition("--")
        frequency_values.extend(
            float(value.replace("D", "E").replace("d", "e"))
            for value in re.findall(number_pattern, frequency_text)
        )
    imaginary_frequency_count = (
        int(nimag_values[-1])
        if nimag_values
        else sum(value < 0.0 for value in frequency_values)
    )
    return {{
        "scf_energy_hartree": (
            float(scf_values[-1].replace("D", "E").replace("d", "e"))
            if scf_values
            else None
        ),
        "free_energy_hartree": (
            float(free_values[-1].replace("D", "E").replace("d", "e"))
            if free_values
            else None
        ),
        "normal_termination": "Normal termination of Gaussian" in text and "Error termination" not in text,
        "optimization_completed": "Optimization completed" in text,
        "imaginary_frequency_count": imaginary_frequency_count if frequency_values or nimag_values else None,
    }}

parsed = {{name: parse_gaussian_log(text) for name, text in raw_logs.items()}}
labels = ("TS-2a", "TS-2b-1")
stages = ("step4_opt_freq", "step10_single_point")
records = {{}}
errors = []
for label in labels:
    prefix = label.lower().replace("-", "_")
    records[label] = {{}}
    for stage in stages:
        key = f"{{prefix}}_{{stage}}"
        item = parsed.get(key)
        if item is None:
            errors.append(f"missing fresh log: {{key}}")
            continue
        records[label][stage] = item
    opt = records[label].get("step4_opt_freq", {{}})
    sp = records[label].get("step10_single_point", {{}})
    if not opt.get("normal_termination") or not opt.get("optimization_completed"):
        errors.append(f"Step 4 did not terminate with a completed optimization: {{label}}")
    if opt.get("imaginary_frequency_count") != 1:
        errors.append(f"Step 4 is not a first-order saddle point: {{label}}")
    if opt.get("free_energy_hartree") is None or opt.get("scf_energy_hartree") is None:
        errors.append(f"missing Step 4 energy: {{label}}")
    if not sp.get("normal_termination") or sp.get("scf_energy_hartree") is None:
        errors.append(f"missing Step 10 energy: {{label}}")

corrected = {{}}
if not errors:
    for label in labels:
        opt = records[label]["step4_opt_freq"]
        sp = records[label]["step10_single_point"]
        corrected[label] = opt["free_energy_hartree"] + (
            sp["scf_energy_hartree"] - opt["scf_energy_hartree"]
        )
ordered = sorted(corrected.items(), key=lambda item: item[1])
delta = None
ee = None
if len(ordered) >= 2:
    delta = (ordered[1][1] - ordered[0][1]) * 627.509474
    ratio = math.exp(-abs(delta) / (0.001987204258 * 303.15))
    ee = (1.0 - ratio) / (1.0 + ratio) * 100.0
result = {{
    "status": "completed" if not errors and delta is not None else "insufficient_fresh_evidence",
    "source": "fresh_native_gaussian_logs",
    "formula": "G_corr_Ha = G_step4_Ha + (E_step10_Ha - E_step4_Ha); delta and ee at 303.15 K",
    "units": {{"energies": "Hartree", "delta_delta_g": "kcal/mol"}},
    "temperature_kelvin": 303.15,
    "parsed_native_records": records,
    "corrected_free_energies_hartree": corrected,
    "lowest_two_labels": [item[0] for item in ordered[:2]],
    "delta_delta_g_kcal_mol": delta,
    "predicted_ee_percent": ee,
    "errors": errors,
}}
ctx.write_json("delta_g_result", result)
'''


def build_case1_fresh_analysis_request(
    input_paths: dict[str, str],
    *,
    script_path: str = "code/case1_fresh_delta_g.py",
) -> dict[str, Any]:
    """构造使用四份 Step 4/10 fresh native 日志的分析请求。"""

    if not input_paths:
        raise ValueError("fresh analysis 请求至少需要一个日志输入")
    return {
        "runtime": "core",
        "script_path": script_path,
        "script_target": script_path,
        "inputs": [
            {
                "name": name,
                "source_path": path,
                "semantic_type": "GaussianLog",
            }
            for name, path in sorted(input_paths.items())
        ],
        "outputs": [
            {
                "name": "delta_g_result",
                "path": "outputs/case1_fresh_delta_g_result.json",
                "semantic_type": "Case1FreshDeltaGResult",
                "media_type": "application/json",
                "required": True,
            }
        ],
        "resource_limits": {"cpu_cores": 1, "memory_mb": 2048, "gpu_count": 0},
        "label": "Case 1 Step 11 fresh native Gaussian delta-G and ee analysis",
    }


def collect_case1_fresh_analysis_inputs(
    runs: dict[str, Any],
    labels: tuple[str, str] = ("TS-2a", "TS-2b-1"),
    operation_root: Path | None = None,
) -> dict[str, str] | None:
    """仅当两条路径的 Step 4/10 都有有效终态日志时返回分析输入。"""

    input_paths: dict[str, str] = {}
    seen_paths: set[str] = set()
    seen_operation_ids: set[str] = set()
    seen_job_ids: set[str] = set()
    seen_checkpoint_paths: set[str] = set()
    for label in labels:
        prefix = label.lower().replace("-", "_")
        for stage in ("step4_opt_freq", "step10_single_point"):
            result = runs.get(f"native_{label}_{stage}")
            if not check_native_job_scientific_gate(result):
                return None
            identity = validate_case1_native_stage_identity(
                result,
                operation_root=operation_root,
            )
            if not identity["valid"]:
                return None
            operation_id = identity["operation_id"]
            job_id = identity["job_id"]
            checkpoint_path = identity["checkpoint_path"]
            if (
                not isinstance(operation_id, str)
                or operation_id in seen_operation_ids
                or not isinstance(job_id, str)
                or job_id in seen_job_ids
                or not isinstance(checkpoint_path, str)
                or checkpoint_path in seen_checkpoint_paths
            ):
                return None
            seen_operation_ids.add(operation_id)
            seen_job_ids.add(job_id)
            seen_checkpoint_paths.add(checkpoint_path)
            log_path = extract_native_log_path(result)
            if not log_path:
                return None
            if log_path in seen_paths:
                return None
            seen_paths.add(log_path)
            input_paths[f"{prefix}_{stage}"] = log_path
    return input_paths


def evaluate_case1_full_native_path(
    runs: dict[str, Any],
    labels: tuple[str, str] = ("TS-2a", "TS-2b-1"),
    operation_root: Path | None = None,
) -> dict[str, Any]:
    """独立检查完整 Case 1 路径，不把 IRC 混入 Step 11 的数值依赖。

    输入是本次运行记录。函数逐项核对两条路径的 Step 4、7、10 是否通过
    MiniChem 机械科学判据并具有终态日志，同时核对下游是否声明同一 Step 4
    job/checkpoint；如果收据提供 IRC 正反向或端点字段，也独立核对连通性。
    函数只返回完整性状态和未通过项，不提交作业，也不改变已有记录。Step 11
    即使能够计算 ΔΔG/ee，IRC 未通过时这里仍返回 ``incomplete``。
    """

    required_stages = ("step4_opt_freq", "step7_irc", "step10_single_point")
    expected_intents = {
        "step4_opt_freq": "transition_state",
        "step7_irc": "reaction_path",
        "step10_single_point": "single_point",
    }
    stage_results: dict[str, dict[str, Any]] = {}
    incomplete_stages: list[str] = []
    seen_operation_ids: set[str] = set()
    seen_job_ids: set[str] = set()
    seen_checkpoint_paths: set[str] = set()
    seen_log_paths: set[str] = set()
    for label in labels:
        step4_result = runs.get(f"native_{label}_step4_opt_freq")
        for stage in required_stages:
            key = f"native_{label}_{stage}"
            result = runs.get(key)
            mechanically_valid = check_native_job_scientific_gate(result)
            log_path = extract_native_log_path(result) if isinstance(result, dict) else None
            intent = result.get("calculation_intent") if isinstance(result, dict) else None
            intent_status = "validated" if intent == expected_intents[stage] else "mismatch"
            identity = validate_case1_native_stage_identity(
                result,
                operation_root=operation_root,
            )
            identity_errors = list(identity["errors"])
            operation_id = identity.get("operation_id")
            job_id = identity.get("job_id")
            checkpoint_path = identity.get("checkpoint_path")
            if isinstance(operation_id, str) and operation_id in seen_operation_ids:
                identity_errors.append("duplicate_operation_id")
            elif isinstance(operation_id, str) and operation_id:
                seen_operation_ids.add(operation_id)
            if isinstance(job_id, str) and job_id in seen_job_ids:
                identity_errors.append("duplicate_job_id")
            elif isinstance(job_id, str) and job_id:
                seen_job_ids.add(job_id)
            if isinstance(checkpoint_path, str) and checkpoint_path in seen_checkpoint_paths:
                identity_errors.append("duplicate_checkpoint_path")
            elif isinstance(checkpoint_path, str) and checkpoint_path:
                seen_checkpoint_paths.add(checkpoint_path)
            if isinstance(log_path, str) and log_path in seen_log_paths:
                identity_errors.append("duplicate_log_path")
            elif isinstance(log_path, str) and log_path:
                seen_log_paths.add(log_path)
            dependency_status = "not_applicable"
            if stage in {"step7_irc", "step10_single_point"}:
                dependency_status = assess_case1_native_dependency_link(
                    step4_result if isinstance(step4_result, dict) else None,
                    result if isinstance(result, dict) else None,
                )
            dependency_unproven = bool(
                stage in {"step7_irc", "step10_single_point"}
                and dependency_status == "not_recorded"
                and isinstance(step4_result, dict)
                and (
                    step4_result.get("job_id")
                    or extract_native_checkpoint_path(step4_result)
                )
            )
            irc_connectivity_status = (
                assess_case1_irc_connectivity_evidence(result)
                if stage == "step7_irc"
                else "not_applicable"
            )
            complete = bool(
                mechanically_valid
                and not identity_errors
                and intent_status != "mismatch"
                and (
                    stage == "step4_opt_freq"
                    or dependency_status == "validated"
                )
                and not dependency_unproven
                and (
                    stage != "step7_irc"
                    or irc_connectivity_status == "validated"
                )
            )
            stage_results[key] = {
                "status": result.get("status") if isinstance(result, dict) else "missing",
                "minichem_status": (
                    result.get("minichem_status") if isinstance(result, dict) else None
                ),
                "mechanically_valid": mechanically_valid,
                "terminal_log_collected": bool(log_path),
                "identity_valid": not identity_errors,
                "identity_errors": identity_errors,
                "operation_id": operation_id,
                "job_id": job_id,
                "checkpoint_path": checkpoint_path,
                "log_path": log_path,
                "calculation_intent": intent,
                "intent_status": intent_status,
                "dependency_link_status": dependency_status,
                "dependency_unproven": dependency_unproven,
                "irc_connectivity_status": irc_connectivity_status,
                "complete": complete,
            }
            if not complete:
                incomplete_stages.append(key)
    scientifically_complete = not incomplete_stages
    return {
        "status": "completed" if scientifically_complete else "incomplete",
        "scientifically_complete": scientifically_complete,
        "required_stages": list(required_stages),
        "stage_results": stage_results,
        "incomplete_stages": incomplete_stages,
        "step11_dependency_stages": ["step4_opt_freq", "step10_single_point"],
        "irc_role": "独立验证过渡态与反应物/产物的连通性，不参与 ΔΔG/ee 数值公式。",
        "identity_policy": (
            "每个已完成 stage 必须有唯一 operation_id/job_id、存在且可核验的"
            " operation receipt、checkpoint 和日志；下游还必须匹配上游 operation/job/checkpoint。"
        ),
    }


def run_case1_fresh_analysis(
    *,
    adapter: MiniChemMcpAdapter,
    runs: dict[str, Any],
    save_manifest: Any,
    operation_root: Path,
    paper_reference_delta_kcal_mol: float | None = None,
) -> dict[str, Any]:
    """在四份 Step 4/10 fresh native 日志齐备后运行 Step 11 分析。"""

    analysis_key = "step11_python_delta_g_analysis"
    operation_id = "case1-step11-fresh-delta-g-analysis"
    existing = runs.get(analysis_key)
    if isinstance(existing, dict):
        if existing.get("status") == "blocked_dependency" and not existing.get("submission_attempted"):
            pass
        else:
            recovered = recover_existing_program_operation(
                adapter=adapter,
                runs=runs,
                key=analysis_key,
                save_manifest=save_manifest,
                operation_root=operation_root,
            )
            return attach_case1_analysis_outcome(
                adapter,
                recovered,
                paper_reference_delta_kcal_mol,
            )

    input_paths = collect_case1_fresh_analysis_inputs(
        runs,
        operation_root=operation_root,
    )
    if input_paths is None:
        blocked = dict(existing) if isinstance(existing, dict) else {}
        blocked.update(
            {
                "status": "blocked_dependency",
                "operation_id": operation_id,
                "submission_attempted": False,
                "blocked_reason": "Step 11 需要两条路径 Step 4/10 的 fresh native stdout.log；当前至少一个数值依赖尚未 completed、未通过机械科学判据或未收集日志。",
                "source": "fresh_native_gaussian_logs_only",
            }
        )
        runs[analysis_key] = blocked
        save_manifest()
        return blocked

    input_names = sorted(input_paths)
    analysis_program = build_case1_fresh_analysis_program(input_names)
    analysis_request = build_case1_fresh_analysis_request(input_paths)
    request_fingerprint = calculate_action_request_fingerprint(analysis_request)
    attempt = {
        "status": "preparing",
        "operation_id": operation_id,
        "request_fingerprint": request_fingerprint,
        "source": "fresh_native_gaussian_logs_only",
        "input_paths": input_paths,
        "script_path": analysis_request["script_path"],
        "submission_attempted": False,
        "operation_receipt_path": str(operation_root / f"{operation_id}.json"),
    }
    runs[analysis_key] = attempt
    save_manifest()
    try:
        adapter.ensure_workspace_text(analysis_request["script_path"], analysis_program)
        validate_analysis = getattr(adapter, "validate_analysis_program", None)
        if callable(validate_analysis):
            attempt["status"] = "validating"
            save_manifest()
            validation = validate_analysis(analysis_request, operation_id)
            attempt["analysis_validation"] = summarize_native_validation(validation)
            if validation.get("status") != "success" or validation.get("valid") is False:
                attempt["status"] = "analysis_validation_failed"
                save_manifest()
                return attempt
        attempt["status"] = "submitting"
        attempt["submission_attempted"] = True
        save_manifest()
        submitted = adapter.submit_analysis_program(analysis_request, operation_id)
        if submitted.get("status") == "waiting":
            submitted = adapter.resume_operation(operation_id, collect=True)
        result = summarize_mcp_result(submitted)
        result.update(
            {
                "operation_id": operation_id,
                "request_fingerprint": request_fingerprint,
                "source": "fresh_native_gaussian_logs_only",
                "input_paths": input_paths,
                "script_path": analysis_request["script_path"],
                "submission_attempted": True,
            }
        )
        if "analysis_validation" in attempt:
            result["analysis_validation"] = attempt["analysis_validation"]
        result = attach_case1_analysis_outcome(
            adapter,
            result,
            paper_reference_delta_kcal_mol,
        )
        runs[analysis_key] = result
    except Exception as exc:  # pragma: no cover - live service diagnostic
        attempt.update(
            {
                "status": "call_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        runs[analysis_key] = attempt
    save_manifest()
    return runs[analysis_key]


def attach_case1_analysis_outcome(
    adapter: MiniChemMcpAdapter,
    result: dict[str, Any] | None,
    paper_reference_delta_kcal_mol: float | None,
) -> dict[str, Any] | None:
    """读取 Step 11 结果，区分程序完成与科学证据通过并附加论文比较。

    MiniChem 返回的 ``completed`` 只说明分析程序正常执行。函数还会读取程序
    生成的 JSON，把能量和频率检查形成的内部状态单独写入 manifest；只有
    Step 11 内部状态也是 ``completed`` 时才比较论文数值。完整路径的双向 IRC
    状态由 ``evaluate_case1_full_native_path`` 独立保存。读取或解析失败会保存
    结构化错误，但不会把论文值用于补齐本次计算结果。
    """

    if not isinstance(result, dict) or result.get("status") != "completed":
        return result
    output_path = next(
        (
            path
            for path in extract_native_output_paths(result)
            if Path(path).name == "case1_fresh_delta_g_result.json"
        ),
        None,
    )
    if output_path is None:
        result["case1_analysis_status"] = "analysis_output_missing"
        result["case1_scientifically_valid"] = False
        return result
    try:
        payload = adapter.read_workspace_text(output_path)
        content = payload.get("content") if isinstance(payload, dict) else None
        if not isinstance(content, str):
            raise ValueError("MiniChem 没有返回 Step 11 JSON 文本")
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("Step 11 输出不是 JSON 对象")
        analysis_status = parsed.get("status")
        result["case1_analysis_status"] = analysis_status
        result["case1_scientifically_valid"] = analysis_status == "completed"
        result["analysis_result"] = parsed
        fresh_delta = parsed.get("delta_delta_g_kcal_mol")
        if (
            analysis_status == "completed"
            and isinstance(fresh_delta, (int, float))
            and paper_reference_delta_kcal_mol is not None
        ):
            result["paper_comparison"] = {
                "fresh_delta_delta_g_kcal_mol": float(fresh_delta),
                "reported_delta_delta_g_kcal_mol": float(paper_reference_delta_kcal_mol),
                "absolute_difference_kcal_mol": abs(
                    float(fresh_delta) - float(paper_reference_delta_kcal_mol)
                ),
                "paper_value_used_only_for_comparison": True,
            }
    except Exception as exc:  # pragma: no cover - live service diagnostic
        result["case1_analysis_status"] = "analysis_output_unreadable"
        result["case1_scientifically_valid"] = False
        result["analysis_result_error"] = {
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return result


def _format_service_result_path(path: Path, runtime_root: Path) -> str:
    """返回适合写入 manifest 的结果路径，不要求测试目录位于仓库内。"""

    for parent in (ROOT, runtime_root):
        try:
            return str(path.relative_to(parent))
        except ValueError:
            continue
    return str(path)


def _check_trace_request_fingerprint(
    trace_record: dict[str, Any],
    *,
    action_id: str,
    expected_fingerprint: str,
) -> bool:
    """核对 MCP trace 中的规范化请求是否对应调用方原始请求。"""

    if trace_record.get("tool") != action_id:
        return False
    arguments = trace_record.get("arguments")
    request = arguments.get("request") if isinstance(arguments, dict) else None
    if not isinstance(request, dict):
        return False
    candidates = [dict(request)]
    with_action = dict(request)
    with_action.setdefault("action_id", action_id)
    candidates.append(with_action)
    # MiniChem 的 MCP 入口会为省略字段补入这两个传输默认值。原始请求指纹
    # 在默认值补齐前计算，因此只允许逆向移除这两个已知默认值。
    authored_shape = dict(with_action)
    if authored_shape.get("component_backends") == {}:
        authored_shape.pop("component_backends")
    if authored_shape.get("source_id") is None:
        authored_shape.pop("source_id")
    candidates.append(authored_shape)
    return any(
        calculate_action_request_fingerprint(candidate) == expected_fingerprint
        for candidate in candidates
    )


def restore_validation_incidents(
    manifest: dict[str, Any],
    *,
    runtime_root: Path | None = None,
) -> None:
    """按请求指纹从 MiniChem trace 恢复 Step 1 结果和重复提交事故。"""

    runtime_root = (
        runtime_root or ROOT / "local_data" / "runtime" / "case1_harness" / "minichem"
    )
    result_root = runtime_root / "_tool_results"
    runs = manifest.setdefault("executions", {})
    official = runs.setdefault("step1_official_composite_pipeline", {})
    fingerprint = official.get("request_fingerprint")
    locator = official.get("result_locator")
    action_id = (
        locator.get("action_id")
        if isinstance(locator, dict)
        else "search_transition_state_path"
    )
    backend_id = locator.get("backend_id") if isinstance(locator, dict) else None
    trace_path = runtime_root / "_tool_trace.jsonl"
    if (
        not result_root.is_dir()
        or not trace_path.is_file()
        or not isinstance(fingerprint, str)
        or not fingerprint
        or not isinstance(action_id, str)
        or not action_id
    ):
        return

    matching_records: list[dict[str, Any]] = []
    try:
        trace_lines = trace_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in trace_lines:
        try:
            record = json.loads(line)
        except (TypeError, ValueError):
            continue
        if not isinstance(record, dict) or not _check_trace_request_fingerprint(
            record,
            action_id=action_id,
            expected_fingerprint=fingerprint,
        ):
            continue
        sequence = record.get("sequence")
        result_path_value = record.get("result_path")
        if not isinstance(sequence, int) or not isinstance(result_path_value, str):
            continue
        result_path = (runtime_root / result_path_value).resolve()
        try:
            result_path.relative_to(runtime_root.resolve())
        except ValueError:
            continue
        matching_records.append(
            {"sequence": sequence, "trace": record, "path": result_path}
        )
    matching_records.sort(key=lambda item: item["sequence"])

    evidence: list[dict[str, Any]] = []
    recoverable_results: list[dict[str, Any]] = []
    for record in matching_records:
        path = record["path"]
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        nested = raw.get("result") if isinstance(raw, dict) else None
        if not isinstance(nested, dict):
            continue
        minichem_status = str(nested.get("status", "")).lower()
        if minichem_status not in {"success", "partial_success"}:
            continue
        if nested.get("action") != action_id:
            continue
        returned_backends = (nested.get("requested_backend"), nested.get("backend"))
        if backend_id is not None and any(
            value not in {None, backend_id} for value in returned_backends
        ):
            continue
        nested_result = nested.get("result") if isinstance(nested, dict) else None
        artifact_id = nested_result.get("artifact_id") if isinstance(nested_result, dict) else None
        evidence.append(
            {
                "sequence": record["sequence"],
                "path": _format_service_result_path(path, runtime_root),
                "status": raw.get("status"),
                "action": nested.get("action"),
                "backend": nested.get("backend"),
                "artifact_id": artifact_id,
                "recorded_at": path.stat().st_mtime,
            }
        )
        recoverable_results.append(nested)
    if not evidence:
        return

    if official.get("status") in {
        "in_progress",
        "call_failed",
        "uncertain_requires_manual_recovery",
    }:
        preserved = {
            name: official.get(name)
            for name in (
                "request_fingerprint",
                "result_locator",
                "operation_id_supported",
                "deterministic_operation_label",
            )
            if name in official
        }
        restored = summarize_mcp_result(recoverable_results[0])
        restored["status"] = "completed"
        restored["minichem_status"] = str(
            recoverable_results[0].get("status", "")
        ).lower()
        restored.update(preserved)
        restored["recovered_from_service_trace"] = True
        restored["recovery_note"] = (
            "已用相同请求指纹的最早 trace/result 恢复；没有再次 execute_action。"
        )
        runs["step1_official_composite_pipeline"] = restored
        official = restored
    official["service_result_evidence"] = evidence
    official.setdefault("first_submitted_result", evidence[0])
    incidents = manifest.setdefault("validation_incidents", [])
    incidents_by_id = {
        item.get("incident_id"): item
        for item in incidents
        if isinstance(item, dict) and isinstance(item.get("incident_id"), str)
    }
    for index, item in enumerate(evidence[1:], start=2):
        incident_id = f"official-composite-duplicate-{Path(item['path']).stem}"
        incident = incidents_by_id.get(incident_id)
        if incident is not None:
            incident.update(
                {
                    "attempt_number": index,
                    "reason": "误用 --force 重复提交同步 Action；正式记录采用相同请求指纹下 trace 顺序最早的服务结果。",
                    "service_result": item,
                    "recovery": "重复结果只保留为事故证据；后续运行不再自动重提。",
                }
            )
            continue
        incident = {
            "incident_id": incident_id,
            "type": "accidental_duplicate_submission",
            "scope": "step1_official_composite_pipeline",
            "attempt_number": index,
            "reason": "误用 --force 重复提交同步 Action；正式记录采用相同请求指纹下 trace 顺序最早的服务结果。",
            "service_result": item,
            "recovery": "重复结果只保留为事故证据；后续运行不再自动重提。",
        }
        incidents.append(incident)
        incidents_by_id[incident_id] = incident

    for incident in incidents:
        if not isinstance(incident, dict):
            continue
        if (
            incident.get("type") == "uncertain_sync_action"
            and incident.get("scope") == "step1_official_composite_pipeline"
        ):
            incident["resolution_status"] = "resolved_from_matching_trace"
            incident["resolution"] = (
                "已用请求指纹匹配最早的 MiniChem trace/result，正式状态恢复为 completed；没有再次提交。"
            )

    probe_restore = {
        "build_reactive_complex_ensemble": "step1_reactive_complex_contract_probe",
        "locate_transition_state": "step1_locate_ts_contract_probe",
    }
    for action_suffix, execution_key in probe_restore.items():
        probe_paths = sorted(
            result_root.glob(f"*_{action_suffix}.json"),
            key=lambda path: path.stat().st_mtime,
        )
        if execution_key in runs or not probe_paths:
            continue
        path = probe_paths[-1]
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        runs[execution_key] = {
            "status": raw.get("status", "unknown"),
            "minichem_status": raw.get("status"),
            "service_result_path": _format_service_result_path(path, runtime_root),
            "service_result_error": raw.get("error"),
            "recovered_from_service_receipt": True,
            "recovery_note": "已存在同步 Action 服务结果；本次运行不再次 execute_action。",
        }


def recover_existing_program_operation(
    *,
    adapter: MiniChemMcpAdapter,
    runs: dict[str, Any],
    key: str,
    save_manifest: Any,
    operation_root: Path,
) -> dict[str, Any]:
    """恢复已有 programmable operation；没有 receipt 时只标人工恢复。"""

    existing = runs.get(key)
    if not isinstance(existing, dict):
        return existing
    operation_id = existing.get("operation_id")
    receipt_path = (
        operation_root / f"{operation_id}.json"
        if isinstance(operation_id, str)
        else None
    )
    status = existing.get("status")
    if isinstance(operation_id, str) and (
        status in {"waiting", "completed"}
        or (receipt_path is not None and receipt_path.is_file())
    ):
        try:
            resumed = adapter.resume_operation(operation_id, collect=True)
            preserved = {
                name: existing.get(name)
                for name in (
                    "request_fingerprint",
                    "source",
                    "input_paths",
                    "script_path",
                    "submission_attempted",
                )
                if name in existing
            }
            runs[key] = summarize_mcp_result(resumed)
            runs[key].update(preserved)
            runs[key]["recovered_without_resubmission"] = True
        except Exception as exc:  # pragma: no cover - live service diagnostic
            existing.update(
                {
                    "status": "recovery_failed",
                    "recovery_without_resubmission": True,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    elif status in {"submitting", "in_progress"}:
        existing["status"] = "uncertain_requires_manual_recovery"
        existing["recovery_required"] = True
        existing.setdefault(
            "recovery_note",
            "已有 analysis 尝试但没有 operation receipt；未再次提交。",
        )
    save_manifest()
    return runs[key]


def build_case1_completeness_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    """分别汇总工具能力完备性和封装粒度完备性，并固定回放边界。"""

    service_status = manifest.get("service", {}).get("status")
    replay = manifest.get("replay", {})
    full_validation = replay.get("full_path_validation", {})
    scientifically_complete = full_validation.get("scientifically_complete") is True
    contracts = manifest.get("discovery", {}).get("candidate_actions", {})
    if not isinstance(contracts, dict):
        contracts = {}
    unresolved_contracts: list[dict[str, Any]] = []
    for step in STATIC_STEPS:
        for action_id in step.get("candidate_actions", []):
            action_contract = contracts.get(action_id)
            if not isinstance(action_contract, dict) or action_contract.get("status") != "catalogued":
                unresolved_contracts.append(
                    {
                        "step": step["step"],
                        "action_id": action_id,
                        "status": (
                            action_contract.get("status")
                            if isinstance(action_contract, dict)
                            else "not_discovered"
                        ),
                    }
                )

    deck_builder_steps = sorted(
        int(step)
        for step in replay.get("deck_builder_coverage", {})
        if str(step).isdigit()
    )
    native_plan = replay.get("native_plan", [])
    native_stages = sorted(
        {
            item.get("stage")
            for item in native_plan
            if isinstance(item, dict) and item.get("stage")
        }
    )
    tool_status = "not_assessed"
    if service_status == "connected":
        # 公开 final-TS 回放即使完整，也不证明从原始 SMILES 独立发现 TS；
        # 因此这里用 partial，避免把受限回放写成完整论文复现。
        tool_status = "partial" if scientifically_complete else "incomplete"
    granularity_status = "partial" if deck_builder_steps else "not_assessed"
    return {
        "tool_capability_completeness": {
            "status": tool_status,
            "scope": "MiniChem Action/native 软件能否取得 Case 1 所需观测；不替代科学结论。",
            "supplied_final_ts_replay_scientifically_complete": scientifically_complete,
            "unresolved_action_contracts": unresolved_contracts,
            "native_stage_evidence": full_validation,
            "raw_smiles_independent_discovery": {
                "status": "not_validated",
                "reason": "论文未公开 TS 初猜、接触原子和构象约束；本脚本不从原始 SMILES 独立生成论文 TS。",
            },
        },
        "encapsulation_granularity_completeness": {
            "status": granularity_status,
            "scope": "是否为论文每个工具步骤提供独立 Action；与底层软件能否执行是两个维度。",
            "dedicated_action_steps": [
                step["step"] for step in STATIC_STEPS if step.get("candidate_actions")
            ],
            "shared_native_deck_builder_steps": deck_builder_steps,
            "native_replay_stages": native_stages,
            "steps_without_dedicated_action": [
                step["step"] for step in STATIC_STEPS if not step.get("candidate_actions")
            ],
            "explanation": (
                "GenerateGaussianCode/xyz_to_gjf 等论文步骤由确定性 Gaussian deck builder "
                "和 open_execution native stage 覆盖；这说明底层路径可组合，但不等于目录中存在逐步同名 Action。"
            ),
        },
        "replay_boundary": {
            "supplied_final_ts_coordinates": True,
            "supplied_final_ts_labels": ["TS-2a", "TS-2b-1"],
            "raw_smiles_independent_discovery": False,
            "statement": (
                "本次 native 回放使用 supplementary/case1.json 的公开 TS final coordinates；"
                "它验证后续工具路径，不应称为从反应物 SMILES 独立复现 TS 初猜或完整原始流水线。"
            ),
        },
    }


def run_mcp_action_once(
    *,
    adapter: MiniChemMcpAdapter,
    runs: dict[str, Any],
    key: str,
    request: dict[str, Any],
    save_manifest: Any,
) -> dict[str, Any]:
    """执行一次同步 Action；已有 key 直接返回，永不因重跑重复调用。"""

    if key in runs:
        return runs[key]
    fingerprint = calculate_action_request_fingerprint(request)
    attempt = {
        "status": "in_progress",
        "attempted_at": time.time(),
        "request_fingerprint": fingerprint,
        "result_locator": {
            "tool": "execute_action",
            "action_id": request.get("action_id"),
            "backend_id": request.get("backend_id"),
        },
    }
    runs[key] = attempt
    save_manifest()
    try:
        runs[key] = summarize_mcp_result(adapter.execute_action(request))
        runs[key].setdefault("request_fingerprint", fingerprint)
    except Exception as exc:  # pragma: no cover - live service diagnostic
        runs[key] = {
            **attempt,
            "status": "call_failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    save_manifest()
    return runs[key]


def run_case1_tool_path_validation(
    endpoint: str,
    output_dir: Path,
    *,
    force: bool = False,
    retry_failed_native: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_existed = manifest_path.exists()
    if manifest_existed:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "schema_version": 1,
            "started_at": time.time(),
            "endpoint": endpoint,
            "paper": {
                "workflow_steps": STATIC_STEPS,
                "case1_input": CASE1_REACTANTS,
                "reference_results": load_case1_summary(),
                "reference_claims": {
                    "temperature_kelvin": 303.15,
                    "opt_method": "B3LYP/6-31G(d)",
                    "single_point_method": "M062X/def2TZVPP",
                    "reported_free_energy_difference_kcal_mol": 1.9,
                    "reported_verified_ts_count": 12,
                },
            },
            "service": {"status": "not_started"},
            "discovery": {},
            "executions": {},
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 同步 Action 没有 operation_id/job_id；即使调用方传入 --force，也不能在
    # 不确定上次结果时自动再次提交。保留兼容参数只是为了避免旧命令报错，并把
    # 这次请求记录为审计事件。
    if force:
        manifest.setdefault("validation_incidents", []).append(
            {
                "type": "force_ignored",
                "time": time.time(),
                "reason": "同步 Action 没有可恢复 operation_id/job_id；已存在的 execution key 一律跳过，不重复提交。",
            }
        )
    if manifest_existed:
        restore_validation_incidents(manifest)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    try:
        with MiniChemMcpAdapter(
            endpoint=endpoint,
            operation_root=output_dir / "operations",
        ) as adapter:
            catalog = adapter.discover_catalog()
            tools = adapter.list_tools()
            manifest["service"] = {
                "status": "connected",
                "catalog_version": catalog.catalog_version,
                "catalog_hash": catalog.catalog_hash,
                "action_count": len(catalog.actions),
                "backend_count": len(catalog.backends),
                "mcp_tool_count": len(tools.get("tools", [])) if isinstance(tools, dict) else None,
            }
            manifest["discovery"]["domains"] = adapter._call_tool("list_action_domains", {"include_action_ids": True})
            manifest["discovery"]["searches"] = {
                query: adapter._call_tool(
                    "search_actions",
                    {"query": query, "retrieval_mode": "hybrid", "detail_level": "summary", "limit": 20},
                )
                for query in ("transition state", "IRC", "Gaussian frequency", "thermochemistry")
            }
            action_ids = sorted({item for step in STATIC_STEPS for item in step["candidate_actions"]})
            contracts: dict[str, Any] = {}
            for action_id in action_ids:
                action = next((item for item in catalog.actions if item.get("id") == action_id), None)
                if action is None:
                    contracts[action_id] = {"status": "missing_from_catalog"}
                    continue
                contracts[action_id] = {
                    "status": "catalogued",
                    "backend_ids": action.get("backend_ids", []),
                    "contracts": {},
                }
                for backend_id in action.get("backend_ids", []):
                    try:
                        contracts[action_id]["contracts"][backend_id] = adapter.inspect_action(action_id, backend_id)["action"]
                    except Exception as exc:  # pragma: no cover - live service diagnostic
                        contracts[action_id]["contracts"][backend_id] = {
                            "status": "inspect_failed",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
            manifest["discovery"]["candidate_actions"] = contracts
            try:
                software_inventory = adapter.list_software(native_only=True, available_only=False)
                manifest["discovery"]["software_inventory"] = software_inventory
                software_rows = software_inventory.get("software", [])
                manifest["discovery"]["software_availability"] = {
                    str(row.get("software_id")): {
                        "available": row.get("available"),
                        "interface_smoke_status": row.get("interface_smoke_status"),
                        "scientific_smoke_status": row.get("scientific_smoke_status"),
                        "native_executables": row.get("native_executables", []),
                    }
                    for row in software_rows
                    if isinstance(row, dict) and row.get("software_id")
                }
                manifest["discovery"]["software_inspections"] = {}
                for software_id in ("gaussian", "multiwfn"):
                    try:
                        manifest["discovery"]["software_inspections"][software_id] = adapter.inspect_software(software_id)
                    except Exception as exc:  # pragma: no cover - live service diagnostic
                        manifest["discovery"]["software_inspections"][software_id] = {
                            "status": "inspect_failed",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
            except Exception as exc:  # pragma: no cover - live service diagnostic
                software_inventory = {"status": "call_failed", "error_type": type(exc).__name__, "error": str(exc)}
                manifest["discovery"]["software_inventory"] = software_inventory
                manifest["discovery"]["software_availability"] = {}
                manifest["discovery"]["software_inspections"] = {}

            # 可重复、低成本且不依赖假设 TS 坐标的前置步骤。
            runs = manifest.setdefault("executions", {})
            software_availability = manifest.get("discovery", {}).get("software_availability", {})
            gaussian_available = bool(
                isinstance(software_availability, dict)
                and software_availability.get("gaussian", {}).get("available") is True
            )
            multiwfn_available = bool(
                isinstance(software_availability, dict)
                and software_availability.get("multiwfn", {}).get("available") is True
            )
            manifest.setdefault("replay", {})["runtime_availability"] = {
                "gaussian": gaussian_available,
                "multiwfn": multiwfn_available,
                "checked_before_replay": True,
            }

            def save_manifest() -> None:
                manifest["updated_at"] = time.time()
                manifest_path.write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
                    encoding="utf-8",
                )

            # 同步 Action 在进程中断后不能通过 job 查询确认；保留原状态并显式
            # 标为需要人工恢复，后续运行只读该记录，不会再次 execute_action。
            for execution_key, execution in runs.items():
                if not isinstance(execution, dict) or execution.get("status") != "in_progress":
                    continue
                if execution.get("operation_id") or execution.get("job_id"):
                    continue
                execution.setdefault("status_before_recovery", "in_progress")
                execution["status"] = "uncertain_requires_manual_recovery"
                execution["recovery_required"] = True
                execution.setdefault(
                    "recovery_note",
                    "同步 Action 没有 operation_id/job_id；不得自动重提，需根据服务端结果定位器人工核对。",
                )
            save_manifest()

            if "standardize_reactant" not in runs:
                run_mcp_action_once(
                    adapter=adapter,
                    runs=runs,
                    key="standardize_reactant",
                    request=build_action_request(
                        "standardize_structure",
                        "rdkit",
                        {"structure": {"smiles": CASE1_REACTANTS["p_nitrobenzaldehyde"]}},
                        settings={
                            "largest_fragment": True,
                            "neutralize": False,
                            "canonical_tautomer": False,
                        },
                    ),
                    save_manifest=save_manifest,
                )
            if "generate_3d_reactant" not in runs:
                result = run_mcp_action_once(
                    adapter=adapter,
                    runs=runs,
                    key="generate_3d_reactant",
                    request=build_action_request(
                        "generate_3d_structure",
                        "rdkit",
                        {"molecule": {"smiles": CASE1_REACTANTS["p_nitrobenzaldehyde"]}},
                        settings={"random_seed": 42},
                    ),
                    save_manifest=save_manifest,
                )
                artifacts = result.get("output_artifacts", []) if isinstance(result, dict) else []
                if artifacts and "reactant_structure_artifact" not in runs:
                    runs["reactant_structure_artifact"] = artifacts[0]
                    save_manifest()
            if "generate_conformers_reactant" not in runs:
                run_mcp_action_once(
                    adapter=adapter,
                    runs=runs,
                    key="generate_conformers_reactant",
                    request=build_action_request(
                        "generate_conformer_ensemble",
                        "rdkit_etkdg",
                        {"molecule": {"smiles": CASE1_REACTANTS["p_nitrobenzaldehyde"]}},
                        settings={"num_conformers": 5, "random_seed": 42},
                    ),
                    save_manifest=save_manifest,
                )
            # 低成本能量仅证明 Action/Backend 可调用，不是论文的 Gaussian 结果。
            if "xtb_energy_reactant" not in runs:
                structure = runs.get("reactant_structure_artifact")
                if structure:
                    run_mcp_action_once(
                        adapter=adapter,
                        runs=runs,
                        key="xtb_energy_reactant",
                        request=build_action_request(
                            "calculate_energy",
                            "xtb",
                            {"structure": structure},
                            method={"method": "GFN2-xTB"},
                        ),
                        save_manifest=save_manifest,
                    )
                else:
                    runs["xtb_energy_reactant"] = {
                        "status": "not_executed",
                        "reason": "generate_3d_structure 未返回可交接 ArtifactRef",
                    }

            # Step 1 先走工具箱为 ARCHE 预留的正式 Case 1 composite Action。该请求来自
            # 工具箱 examples，不把论文的 TSPipline 名称硬映射成 generic Action。
            composite_example = (
                ROOT
                / "third_party"
                / "minichem_toolbox"
                / "examples"
                / "arche_23_ts_pipeline_case1.json"
            )
            composite_request = json.loads(composite_example.read_text(encoding="utf-8"))
            if (
                composite_request.get("action_id")
                != "search_transition_state_path"
                or composite_request.get("backend_id") != "transition_state_path_search"
                or composite_request.get("inputs")
                or composite_request.get("method_spec")
                or composite_request.get("action_settings")
                or int((composite_request.get("resource_limits") or {}).get("gpu_count", 0)) < 1
            ):
                raise ValueError(
                    "固定 Case 1 composite request 必须使用空科学字段并申请至少一个 GPU"
                )
            composite_fingerprint = calculate_action_request_fingerprint(composite_request)
            composite_locator = {
                "tool": "execute_action",
                "action_id": composite_request.get("action_id"),
                "backend_id": composite_request.get("backend_id"),
                "request_fingerprint": composite_fingerprint,
                "tool_result_glob": (
                    "local_data/runtime/case1_harness/minichem/"
                    "_tool_results/*_search_transition_state_path.json"
                ),
                "operation_id_supported": False,
                "deterministic_operation_label": "case1-step1-official-composite",
            }
            if "step1_official_composite_pipeline" not in runs:
                runs["step1_official_composite_pipeline"] = {
                    "status": "in_progress",
                    "started_at": time.time(),
                    "request_fingerprint": composite_fingerprint,
                    "result_locator": composite_locator,
                    "operation_id_supported": False,
                    "deterministic_operation_label": "case1-step1-official-composite",
                }
                save_manifest()
                try:
                    runs["step1_official_composite_pipeline"] = summarize_mcp_result(
                        adapter.execute_action(composite_request)
                    )
                    runs["step1_official_composite_pipeline"].setdefault(
                        "request_fingerprint", composite_fingerprint
                    )
                    runs["step1_official_composite_pipeline"]["result_locator"] = composite_locator
                    runs["step1_official_composite_pipeline"]["operation_id_supported"] = False
                except Exception as exc:  # pragma: no cover - live service diagnostic
                    runs["step1_official_composite_pipeline"] = {
                        "status": "call_failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "request_fingerprint": composite_fingerprint,
                        "result_locator": composite_locator,
                    }
                save_manifest()
            else:
                # 已有 completed/partial_success/in_progress/uncertain 记录都只读；
                # 特别是同步 Action 的 in_progress 不能靠 --force 猜测后重提。
                runs["step1_official_composite_pipeline"].setdefault(
                    "result_locator", composite_locator
                )
                runs["step1_official_composite_pipeline"].setdefault(
                    "request_fingerprint", composite_fingerprint
                )
                save_manifest()
            manifest.setdefault("paper", {}).setdefault("workflow_steps", [])[0][
                "official_action"
            ] = "search_transition_state_path/transition_state_path_search"
            save_manifest()

            # Step 1 的两个入口分别做一次真实合同探针：ReactiveComplex 缺论文未给出的
            # forming-contact 定义，因此预期是结构化 invalid_request；TS locator 使用公开
            # 最终坐标只验证执行入口，不声称它是从 SMILES 得到的初猜。
            if "step1_reactive_complex_contract_probe" not in runs:
                # 协议层禁止同一个 ArtifactRef 在列表中重复出现。这里仅提交
                # 一个公开反应物结构和空接触对，验证缺失反应物/成键约束时的
                # 结构化错误；不复制同一 ArtifactRef 冒充多个反应物。
                components = []
                if runs.get("reactant_structure_artifact"):
                    components = [runs["reactant_structure_artifact"]]
                run_mcp_action_once(
                    adapter=adapter,
                    runs=runs,
                    key="step1_reactive_complex_contract_probe",
                    request=build_action_request(
                        "build_reactive_complex_ensemble",
                        "internal_reactive_complex",
                        {"components": components, "contact_pairs": []},
                        settings={
                            "random_seed": 42,
                            "candidate_count": 1,
                            "max_candidates": 1,
                            "orientation_trials": 4,
                            "contact_tolerance_angstrom": 0.2,
                            "minimum_interatomic_distance_angstrom": 0.8,
                        },
                    ),
                    save_manifest=save_manifest,
                )
                runs["step1_reactive_complex_contract_gap"] = (
                    "论文只给 SMILES，不给接触原子/目标距离；空 contact_pairs 不能构造反应复合物。"
                )
                save_manifest()

            if "step1_locate_ts_contract_probe" not in runs:
                case1_data_for_probe = json.loads(CASE1_PATH.read_text(encoding="utf-8"))
                probe_structure = convert_case1_record_to_atomic_structure(case1_data_for_probe["opt"]["TS-2a"])
                run_mcp_action_once(
                    adapter=adapter,
                    runs=runs,
                    key="step1_locate_ts_contract_probe",
                    request=build_action_request(
                        "locate_transition_state",
                        "pysisyphus",
                        {"initial_guess": probe_structure},
                        method={
                            "calculator_backend": "xtb",
                            "method": "GFN2-xTB",
                            "charge": probe_structure["charge"],
                            "multiplicity": probe_structure["multiplicity"],
                        },
                        settings={
                            "optimizer": "rsirfo",
                            "convergence": "never",
                            "max_cycles": 1,
                            "hessian_init": "xtb",
                        },
                    ),
                    save_manifest=save_manifest,
                )
                runs["step1_locate_ts_probe_scope"] = (
                    "使用 case1.json 的公开 TS 最终坐标作为输入探针；该结果不证明论文 Step 1 初猜生成。"
                )
                save_manifest()

            # 下游回放使用附录公开的最终 TS 坐标。它能隔离验证 Steps 4/7/10/11
            # 的工具边界，但不证明 Step 1 能从 SMILES 独立发现这些 TS。
            case1_data = json.loads(CASE1_PATH.read_text(encoding="utf-8"))
            replay_records = {
                label: case1_data["opt"][label]
                for label in ("TS-2a", "TS-2b-1")
            }
            replay_meta = manifest.setdefault("replay", {})
            replay_meta["scope"] = (
                "公开最终 TS 坐标回放；不等价于从 SMILES 生成 TS 初猜，也不替代论文完整 14×opt/SP 批处理。"
            )
            replay_meta["reference_labels"] = list(replay_records)
            replay_meta["reference_freeenergies"] = {
                label: replay_records[label].get("freeenergy") for label in replay_records
            }

            replay_meta["execution_layer"] = "native_software"
            replay_meta["full_case1_record_count"] = {
                "opt": len(case1_data.get("opt", {})),
                "sp": len(case1_data.get("sp", {})),
            }
            replay_meta["full_case1_records_not_run"] = True
            replay_meta["reference_coordinate_scope"] = (
                "仅使用附录 case1.json 的 TS-2a 与 TS-2b-1 final coordinates；"
                "不是从论文反应物 SMILES 独立生成的 TS 初猜。"
            )
            replay_meta["native_input_chaining"] = True
            replay_meta["native_input_chaining_note"] = (
                "每个对映路径先运行 Step4；仅当 Step4 返回 completed 且收集到 .chk 时，"
                "才允许 Step7 IRC 和 Step10 SP staged 同一 checkpoint，并记录 parent_operation_id、"
                "parent_job_id 和 parent_checkpoint_path。"
            )
            replay_meta["native_plan"] = []
            replay_meta["deck_builder_coverage"] = {
                str(step): {
                    "covered_by": "deterministic_gaussian_deck_builder + MCP validate_native_job",
                    "separate_action": False,
                }
                for step in (2, 3, 5, 6, 8, 9)
            }
            for label, record in replay_records.items():
                native_results_by_stage: dict[str, dict[str, Any]] = {}
                for plan in CASE1_NATIVE_REPLAY_PLAN:
                    dependency_result = None
                    dependency_stage = plan.get("depends_on")
                    if dependency_stage:
                        dependency_result = native_results_by_stage.get(str(dependency_stage))
                    result = run_case1_native_step(
                        adapter=adapter,
                        runs=runs,
                        save_manifest=save_manifest,
                        label=label,
                        record=record,
                        plan=plan,
                        gaussian_available=gaussian_available,
                        operation_root=output_dir / "operations",
                        dependency_result=dependency_result,
                        retry_failed=retry_failed_native,
                    )
                    native_results_by_stage[str(plan["stage"])] = result
                    replay_meta["native_plan"].append(
                        {
                            "label": label,
                            "stage": plan["stage"],
                            "paper_steps": list(plan["paper_steps"]),
                            "operation_id": result.get("operation_id") if isinstance(result, dict) else None,
                            "status": result.get("status") if isinstance(result, dict) else None,
                            "route": plan["route"],
                            "depends_on": plan.get("depends_on"),
                            "resource_limits": (
                                result.get("resource_limits")
                                if isinstance(result, dict)
                                else case1_native_resource_limits(plan)
                            ),
                            "parent_operation_id": result.get("parent_operation_id") if isinstance(result, dict) else None,
                            "parent_job_id": result.get("parent_job_id") if isinstance(result, dict) else None,
                            "parent_checkpoint_path": result.get("parent_checkpoint_path") if isinstance(result, dict) else None,
                            "checkpoint_path": result.get("checkpoint_path") if isinstance(result, dict) else None,
                        }
                    )
                    save_manifest()

            # 完整路径仍要求 Step 7 IRC；Step 11 的附录明文输入只包含 Step 4
            # 热化学和 Step 10 电子能，因此二者分开记录，避免 IRC 超时错误阻断
            # 已经具备输入的数值后处理。
            replay_meta["full_path_validation"] = evaluate_case1_full_native_path(
                runs,
                operation_root=output_dir / "operations",
            )
            save_manifest()

            # Step 11 只读取本次 Step 4/10 收集的 fresh native 日志；数值依赖未终态
            # 或缺少日志时保存 blocked_dependency，不提交分析作业，也不读取 paper
            # 解析常数冒充新计算结果。
            run_case1_fresh_analysis(
                adapter=adapter,
                runs=runs,
                save_manifest=save_manifest,
                operation_root=output_dir / "operations",
                paper_reference_delta_kcal_mol=(
                    manifest.get("paper", {})
                    .get("reference_claims", {})
                    .get("reported_free_energy_difference_kcal_mol")
                ),
            )
    except Exception as exc:  # pragma: no cover - live service diagnostic
        manifest["service"] = {
            "status": "unavailable_or_failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    manifest.update(build_case1_completeness_summary(manifest))
    manifest["updated_at"] = time.time()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:9010/mcp",
        help="历史 Case 1 工具路径核验使用的 legacy/general 地址；不是当前 research_task 入口",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="兼容旧命令但不会覆盖或重提任何已有 execution；建议省略",
    )
    parser.add_argument(
        "--retry-failed-native",
        action="store_true",
        help="仅归档并重提已明确终态失败的 native 作业，使用新的 attempt operation_id",
    )
    parser.add_argument(
        "--require-complete-path",
        action="store_true",
        help="要求完整 native 路径通过科学完整性核验，否则返回非零状态",
    )
    args = parser.parse_args()
    result = run_case1_tool_path_validation(
        args.endpoint,
        args.output_dir,
        force=args.force,
        retry_failed_native=args.retry_failed_native,
    )
    print(
        json.dumps(
            {
                "service": result.get("service"),
                "executions": result.get("executions"),
                "tool_capability_completeness": result.get(
                    "tool_capability_completeness"
                ),
                "encapsulation_granularity_completeness": result.get(
                    "encapsulation_granularity_completeness"
                ),
                "replay_boundary": result.get("replay_boundary"),
                "full_path_validation": result.get("replay", {}).get(
                    "full_path_validation"
                ),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    if result.get("service", {}).get("status") != "connected":
        return 2
    if (
        args.require_complete_path
        and result.get("replay", {}).get("full_path_validation", {}).get(
            "scientifically_complete"
        )
        is not True
    ):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
