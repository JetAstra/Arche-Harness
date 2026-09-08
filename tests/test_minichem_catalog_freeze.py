"""MiniChem 目录冻结和恢复的资源政策合同测试。"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from chemistry_multiagent.tools.minichem_catalog_freeze import (
    FrozenMiniChemCatalog,
    MiniChemCatalogFreezeError,
    _validate_managed_program_contract,
    freeze_minichem_catalog,
    load_frozen_minichem_catalog,
    load_frozen_minichem_catalog_startup,
)
from chemistry_multiagent.tools.minichem_adapter import (
    MiniChemCatalogSnapshot,
    MiniChemMcpAdapter,
)
from minichem_toolbox.catalog import public_catalog_snapshot
from tests.fakes.minichem_mcp import build_catalog_snapshot


_ABSOLUTE_INPUT = str(Path(Path.cwd().anchor) / "tmp" / "input.com")


def _canonical_bytes(value: object) -> bytes:
    """按目录协议编码快照，供测试构造可读但预算不批准的包。"""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_catalog_package(task_root: Path, snapshot: dict) -> str:
    """写入一份 hash 完整但资源预算被替换的冻结包。"""

    payload_without_hash = deepcopy(snapshot)
    payload_without_hash.pop("catalog_hash", None)
    catalog_hash = hashlib.sha256(_canonical_bytes(payload_without_hash)).hexdigest()
    payload = deepcopy(payload_without_hash)
    payload["catalog_hash"] = catalog_hash
    snapshot_bytes = _canonical_bytes(payload)
    package_root = task_root / "internal" / "catalog" / catalog_hash
    package_root.mkdir(parents=True)
    (package_root / "catalog_snapshot.json").write_bytes(snapshot_bytes)
    (package_root / "catalog_manifest.json").write_bytes(
        _canonical_bytes(
            {
                "schema_version": 1,
                "catalog_hash": catalog_hash,
                "catalog_version": "7",
                "snapshot_relative_path": "catalog_snapshot.json",
                "snapshot_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
                "snapshot_size": len(snapshot_bytes),
            }
        )
    )
    return catalog_hash


def _open_execution_snapshot() -> dict:
    """构造 adapter 返回的最小 Open Execution schema v1。"""

    return {
        "schema_version": 1,
        "native_software": [
            {
                "software_id": "gaussian",
                "display_name": "Gaussian 16",
                "commands": [
                    {
                        "executable": "g16",
                        "synopsis": "运行一个显式 Gaussian 输入文件。",
                        "input_mode": "stdin_or_staged_input",
                        "required_files": ["input.com"],
                        "example_request": {
                            "software_id": "gaussian",
                            "executable": "g16",
                            "arguments": [],
                            "staged_inputs": [
                                {"source_path": "code/input.com", "target_path": "input.com"}
                            ],
                            "stdin_target": "input.com",
                            "resource_policy": {
                                "mode": "service_managed_full_capacity",
                                "concurrency": "serial",
                                "agent_controllable": False,
                            },
                            "calculation_intent": "frequency",
                            "label": "example",
                            "parent_job_id": None,
                        },
                    }
                ],
            }
        ],
        "analysis_runtimes": [
            {
                "runtime": "arche",
                "description": "ARCHE 标准分析 Python 环境。",
                "modules": ["numpy", "cclib"],
            }
        ],
        "authoring_contract": {
            "path_prefix": "code/",
            "max_files": 4,
            "max_chars_per_file": 100000,
        },
    }


def _real_analysis_template() -> dict:
    """构造 MiniChem 真实 discovery 返回的 Analysis 请求模板。"""

    return {
        "runtime": "<exact runtime id>",
        "script_path": "code/<agent-authored-program>.py",
        "script_target": "code/agent_program.py",
        "arguments": [],
        "staged_inputs": [],
        "inputs": [
            {
                "name": "<logical_input_name>",
                "source_path": "data/<required-input>",
                "target_path": "inputs/<required-input>",
                "semantic_type": "<input semantic type>",
            }
        ],
        "outputs": [
            {
                "name": "<logical_output_name>",
                "path": "outputs/<result-file>",
                "semantic_type": "<output semantic type>",
                "media_type": "application/json",
                "required": True,
            }
        ],
        "label": "<descriptive scientific operation>",
    }


def _open_execution_snapshot_with_real_analysis_template() -> dict:
    """在最小 Open Execution 快照中加入真实 Analysis 模板。"""

    snapshot = _open_execution_snapshot()
    template = _real_analysis_template()
    snapshot["submit_analysis_program_request_template"] = template
    snapshot["evaluation_resource_budget"] = deepcopy(
        {
            "source": "service_policy",
            "agent_controllable": False,
            "scope": "per_task",
        }
    )
    snapshot["execution_timeout_policy"] = {
        "execution_class": "compute",
        "source": "service_policy",
        "agent_controllable": False,
    }
    return snapshot


def _adapter_with_catalog(monkeypatch, open_snapshot: dict | None = None) -> MiniChemMcpAdapter:
    """构造不连接网络的 adapter，并注入确定性的两个发现结果。"""

    adapter = object.__new__(MiniChemMcpAdapter)
    catalog = MiniChemCatalogSnapshot(
        catalog_hash=build_catalog_snapshot()["catalog_hash"],
        catalog_version="7",
        snapshot=build_catalog_snapshot(),
    )
    monkeypatch.setattr(
        MiniChemMcpAdapter,
        "discover_catalog",
        lambda self: catalog,
    )
    monkeypatch.setattr(
        MiniChemMcpAdapter,
        "discover_open_execution_catalog",
        lambda self: deepcopy(open_snapshot if open_snapshot is not None else _open_execution_snapshot()),
        raising=False,
    )
    monkeypatch.setattr(
        MiniChemMcpAdapter,
        "discover_action_contracts",
        lambda self: {
            "schema_version": 1,
            "contracts": [
                {
                    "action_id": "convert_structure",
                    "backend_id": "openbabel",
                    "request_contract": {
                        "required_contract": {
                            "inputs": [{"name": "structure"}],
                            "method_spec": [],
                            "action_settings": [],
                            "resource_limits": [],
                        },
                        "optional_field_names": [],
                        "execute_action_request_template": {
                            "inputs": {"structure": {}},
                            "method_spec": {},
                            "action_settings": {},
                            "resource_limits": {},
                        },
                        "enumerated_options": [],
                        "conditional_requirements": [],
                        "output_contract": {"primary_output": "AtomicStructure"},
                    },
                    "output_contract": {"primary_output": "AtomicStructure"},
                }
            ],
        },
        raising=False,
    )
    return adapter


def _adapter_with_public_catalog(monkeypatch) -> tuple[MiniChemMcpAdapter, dict]:
    """构造任务无关的脱敏 public projection 及其最小 pair 合同。"""

    canonical = build_catalog_snapshot(
        actions=[
            {
                "id": "convert_structure",
                "version": "1.0.0",
                "category": "structure_and_system",
                "description": "转换调用方提供的分子结构。",
                "backend_ids": ["openbabel"],
                "required_inputs": ["structure"],
                "optional_inputs": [],
                "selection_policy": "agent_backend_required",
                "primary_output": "AtomicStructure",
                "data_action": False,
                "batch_safe": False,
            },
            {
                "id": "search_transition_state_path",
                "version": "1.0.0",
                "category": "reaction_and_kinetics",
                "description": "根据调用方提供的反应物和产物搜索过渡态路径候选。",
                "backend_ids": ["transition_state_path_search"],
                "required_inputs": ["reactants", "products"],
                "optional_inputs": ["main_force_field", "target_force_field"],
                "selection_policy": "agent_backend_required",
                "primary_output": "TransitionStateGuessPipelineResult",
                "data_action": False,
                "batch_safe": False,
            },
        ],
        backends=[
            {
                "id": "openbabel",
                "display_name": "Open Babel",
                "runtime": "native",
                "capabilities": ["convert_structure"],
                "health": {"status": "healthy"},
            },
            {
                "id": "transition_state_path_search",
                "display_name": "Transition-State Path Search",
                "runtime": "ash_neb",
                "capabilities": ["search_transition_state_path"],
                "health": {"status": "healthy"},
            },
        ],
    )
    public = public_catalog_snapshot(canonical)
    catalog = MiniChemCatalogSnapshot(
        catalog_hash=public["catalog_hash"],
        catalog_version="7",
        snapshot=public,
    )
    adapter = object.__new__(MiniChemMcpAdapter)
    monkeypatch.setattr(MiniChemMcpAdapter, "discover_catalog", lambda self: catalog)
    monkeypatch.setattr(
        MiniChemMcpAdapter,
        "discover_open_execution_catalog",
        lambda self: _open_execution_snapshot(),
        raising=False,
    )
    monkeypatch.setattr(
        MiniChemMcpAdapter,
        "discover_action_contracts",
        lambda self: {
            "schema_version": 1,
            "catalog_hash": public["catalog_hash"],
            "contracts": [
                {
                    "action_id": "convert_structure",
                    "backend_id": "openbabel",
                    "request_contract": {
                        "input_description": "提供一个明确结构并选择输出格式。",
                        "required_contract": {
                            "inputs": [{"name": "structure"}],
                            "method_spec": [],
                            "action_settings": [],
                            "resource_limits": [],
                        },
                        "optional_field_names": [],
                        "execute_action_request_template": {
                            "inputs": {"structure": {}},
                            "method_spec": {},
                            "action_settings": {},
                            "resource_limits": {},
                        },
                        "enumerated_options": [],
                        "conditional_requirements": [],
                        "evaluation_resource_budget": {
                            "source": "service_policy",
                            "agent_controllable": False,
                            "scope": "per_task",
                        },
                        "output_contract": {
                            "primary_output": "AtomicStructure"
                        },
                    },
                    "output_contract": {
                        "primary_output": "AtomicStructure"
                    },
                },
                {
                    "action_id": "search_transition_state_path",
                    "backend_id": "transition_state_path_search",
                    "request_contract": {
                        "input_description": "提供明确的反应物和产物，并按需选择力场。",
                        "required_contract": {
                            "inputs": [
                                {"name": "reactants"},
                                {"name": "products"},
                            ],
                            "method_spec": [],
                            "action_settings": [],
                            "resource_limits": [],
                        },
                        "optional_field_names": [
                            "inputs.main_force_field",
                            "inputs.target_force_field",
                        ],
                        "execute_action_request_template": {
                            "inputs": {
                                "reactants": [],
                                "products": [],
                            },
                            "method_spec": {},
                            "action_settings": {},
                            "resource_limits": {},
                        },
                        "enumerated_options": [],
                        "conditional_requirements": [],
                        "resource_policy": {
                            "mode": "service_managed_full_capacity",
                            "concurrency": "serial",
                            "agent_controllable": False,
                        },
                        "output_contract": {
                            "primary_output": "TransitionStateGuessPipelineResult"
                        },
                    },
                    "output_contract": {
                        "primary_output": "TransitionStateGuessPipelineResult"
                    },
                }
            ],
        },
        raising=False,
    )
    return adapter, public


def _write_schema2_package(
    task_root: Path,
    snapshot: dict,
    open_snapshot: dict,
) -> str:
    """写入可独立篡改 open snapshot 的 schema v2 冻结包。"""

    payload_without_hash = deepcopy(snapshot)
    payload_without_hash.pop("catalog_hash", None)
    catalog_hash = hashlib.sha256(_canonical_bytes(payload_without_hash)).hexdigest()
    catalog_payload = deepcopy(payload_without_hash)
    catalog_payload["catalog_hash"] = catalog_hash
    catalog_bytes = _canonical_bytes(catalog_payload)
    open_bytes = _canonical_bytes(open_snapshot)
    package_root = task_root / "internal" / "catalog" / catalog_hash
    package_root.mkdir(parents=True)
    (package_root / "catalog_snapshot.json").write_bytes(catalog_bytes)
    (package_root / "open_execution_snapshot.json").write_bytes(open_bytes)
    (package_root / "catalog_manifest.json").write_bytes(
        _canonical_bytes(
            {
                "schema_version": 2,
                "catalog_hash": catalog_hash,
                "catalog_version": "7",
                "snapshot_relative_path": "catalog_snapshot.json",
                "snapshot_sha256": hashlib.sha256(catalog_bytes).hexdigest(),
                "snapshot_size": len(catalog_bytes),
                "open_execution_relative_path": "open_execution_snapshot.json",
                "open_execution_sha256": hashlib.sha256(open_bytes).hexdigest(),
                "open_execution_size": len(open_bytes),
            }
        )
    )
    return catalog_hash


def test_freeze_writes_independent_open_execution_and_action_contract_snapshots(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """冻结时 v7 原文/hash 不变，Open Execution 单独写入并进入内存视图。"""

    adapter = _adapter_with_catalog(monkeypatch)
    expected_catalog = build_catalog_snapshot()
    frozen = freeze_minichem_catalog(adapter=adapter, task_root=tmp_path)

    package_root = tmp_path / "internal" / "catalog" / frozen.catalog_hash
    catalog_bytes = (package_root / "catalog_snapshot.json").read_bytes()
    open_bytes = (package_root / "open_execution_snapshot.json").read_bytes()
    manifest = json.loads((package_root / "catalog_manifest.json").read_text(encoding="utf-8"))

    assert json.loads(catalog_bytes) == expected_catalog
    assert json.loads(open_bytes) == _open_execution_snapshot()
    assert manifest["schema_version"] == 3
    assert manifest["open_execution_relative_path"] == "open_execution_snapshot.json"
    assert manifest["open_execution_sha256"] == hashlib.sha256(open_bytes).hexdigest()
    assert manifest["open_execution_size"] == len(open_bytes)
    contract_bytes = (package_root / "action_contract_snapshot.json").read_bytes()
    assert manifest["action_contract_relative_path"] == "action_contract_snapshot.json"
    assert manifest["action_contract_sha256"] == hashlib.sha256(contract_bytes).hexdigest()
    assert json.loads(contract_bytes)["catalog_hash"] == frozen.catalog_hash
    assert frozen.snapshot["catalog_hash"] == expected_catalog["catalog_hash"]
    assert frozen.snapshot["open_execution"] == _open_execution_snapshot()
    assert frozen.snapshot["action_contracts"]["contracts"][0]["action_id"] == "convert_structure"


def test_public_catalog_projection_freezes_and_resumes_with_projection_hash(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """public projection 冻结/恢复保留 canonical 身份并核对独立 projection hash。"""

    adapter, public = _adapter_with_public_catalog(monkeypatch)
    frozen = freeze_minichem_catalog(adapter=adapter, task_root=tmp_path)
    restored = load_frozen_minichem_catalog(
        task_root=tmp_path,
        catalog_hash=public["catalog_hash"],
    )

    assert frozen.catalog_hash == public["catalog_hash"]
    assert frozen.snapshot["catalog_projection_hash"] == public[
        "catalog_projection_hash"
    ]
    assert restored.catalog_hash == frozen.catalog_hash
    assert restored.snapshot["catalog_visibility_policy"] == (
        "public_contract_projection"
    )
    assert restored.snapshot["catalog_projection_hash"] == public[
        "catalog_projection_hash"
    ]
    request_contract = frozen.snapshot["action_contracts"]["contracts"][0][
        "request_contract"
    ]
    assert request_contract["input_description"] == "提供一个明确结构并选择输出格式。"
    assert request_contract["evaluation_resource_budget"] == {
        "source": "service_policy",
        "agent_controllable": False,
        "scope": "per_task",
    }
    assert frozen.snapshot["action_contracts"]["contracts"][1][
        "request_contract"
    ]["resource_policy"] == {
        "mode": "service_managed_full_capacity",
        "concurrency": "serial",
        "agent_controllable": False,
    }

    snapshot_path = (
        tmp_path
        / "internal"
        / "catalog"
        / public["catalog_hash"]
        / "catalog_snapshot.json"
    )
    tampered = json.loads(snapshot_path.read_text(encoding="utf-8"))
    tampered["actions"][0]["description"] = "tampered"
    tampered_bytes = _canonical_bytes(tampered)
    snapshot_path.write_bytes(tampered_bytes)
    manifest_path = snapshot_path.with_name("catalog_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["snapshot_sha256"] = hashlib.sha256(tampered_bytes).hexdigest()
    manifest["snapshot_size"] = len(tampered_bytes)
    manifest_path.write_bytes(_canonical_bytes(manifest))
    with pytest.raises(MiniChemCatalogFreezeError, match="projection"):
        load_frozen_minichem_catalog(
            task_root=tmp_path,
            catalog_hash=public["catalog_hash"],
        )


def test_startup_catalog_check_only_recomputes_key_hashes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """启动复验核对 Catalog/Action 合同摘要，跳过 Open Execution 文件摘要。"""

    frozen = freeze_minichem_catalog(adapter=_adapter_with_catalog(monkeypatch), task_root=tmp_path)
    package_root = tmp_path / "internal" / "catalog" / frozen.catalog_hash
    open_path = package_root / "open_execution_snapshot.json"
    open_payload = json.loads(open_path.read_text(encoding="utf-8"))
    open_payload["native_software"] = []
    open_path.write_bytes(_canonical_bytes(open_payload))

    # 只做关键摘要的启动检查；Open Execution 摘要漂移留给完整复验发现。
    restored = load_frozen_minichem_catalog_startup(
        task_root=tmp_path,
        catalog_hash=frozen.catalog_hash,
    )
    assert restored.catalog_hash == frozen.catalog_hash
    with pytest.raises(MiniChemCatalogFreezeError, match="Open Execution sha256"):
        load_frozen_minichem_catalog(task_root=tmp_path, catalog_hash=frozen.catalog_hash)


def test_freeze_accepts_real_analysis_discovery_template_placeholders(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """真实 Analysis 模板的运行时和路径占位符应在冻结时原样保留。"""

    open_snapshot = _open_execution_snapshot_with_real_analysis_template()
    adapter = _adapter_with_catalog(monkeypatch, open_snapshot)

    frozen = freeze_minichem_catalog(adapter=adapter, task_root=tmp_path)

    template = frozen.snapshot["open_execution"][
        "submit_analysis_program_request_template"
    ]
    assert template["runtime"] == "<exact runtime id>"
    assert template["script_path"] == "code/<agent-authored-program>.py"
    assert template["inputs"][0]["source_path"] == "data/<required-input>"
    assert template["outputs"][0]["path"] == "outputs/<result-file>"


@pytest.mark.parametrize("runtime", ["<other runtime>", "Runtime 1", ""])
def test_freeze_rejects_unapproved_analysis_runtime_template_text(
    tmp_path: Path,
    monkeypatch,
    runtime: str,
) -> None:
    """除协议规定的占位符外，模板 runtime 不能使用任意说明文本。"""

    open_snapshot = _open_execution_snapshot_with_real_analysis_template()
    open_snapshot["submit_analysis_program_request_template"]["runtime"] = runtime
    adapter = _adapter_with_catalog(monkeypatch, open_snapshot)

    with pytest.raises(MiniChemCatalogFreezeError, match="lower_snake_case"):
        freeze_minichem_catalog(adapter=adapter, task_root=tmp_path)


def test_managed_program_contract_accepts_real_multiline_minichem_template() -> None:
    """真实 MiniChem 返回的七行 JobContext 模板必须能通过严格合同校验。"""

    contract = {
        "job_context_required_for_compliance": True,
        "execution_note": (
            "Execute scientific code with submit_analysis_program, not a built-in shell. "
            "The program starts in an isolated job directory. Declare named inputs and "
            "outputs, and use JobContext helpers whose names exactly match the declarations."
        ),
        "minimal_python_template": (
            "from researchchem_job import JobContext\n"
            "ctx = JobContext.load()\n"
            "# value = ctx.input('declared_input_name').read_text()\n"
            "ctx.write_json('declared_json_output_name', result_payload)\n"
            "ctx.output('declared_table_output_name').write_text(csv_text)\n"
            "ctx.register_output('declared_table_output_name')\n"
        ),
        "helper_name_rule": (
            "JobContext helper names must exactly match the corresponding inputs/outputs "
            "declaration names."
        ),
    }

    validated = _validate_managed_program_contract(contract)

    assert validated == contract
    assert validated["minimal_python_template"].count("\n") == 6


@pytest.mark.parametrize(
    ("template", "message"),
    [
        (" \n\t", "非空"),
        ("template\x00line", "NUL"),
        ("template" * 100_001, "字符上限"),
    ],
)
def test_managed_program_contract_rejects_invalid_multiline_template(
    template: str,
    message: str,
) -> None:
    """多行模板仍拒绝空白、NUL 和异常超长输入。"""

    contract = {
        "job_context_required_for_compliance": True,
        "execution_note": "说明",
        "minimal_python_template": template,
        "helper_name_rule": "规则",
    }

    with pytest.raises(MiniChemCatalogFreezeError, match=message):
        _validate_managed_program_contract(contract)


def test_managed_program_contract_keeps_human_notes_single_line() -> None:
    """模板字段允许换行，但其他说明字段仍必须保持单行。"""

    contract = {
        "job_context_required_for_compliance": True,
        "execution_note": "第一行\n第二行",
        "minimal_python_template": "print('ok')\n",
        "helper_name_rule": "规则",
    }

    with pytest.raises(MiniChemCatalogFreezeError, match="单行"):
        _validate_managed_program_contract(contract)


def test_load_legacy_schema1_adds_empty_open_execution_contract(tmp_path: Path) -> None:
    """旧任务没有第二个文件时仍可读取，但只暴露安全的空 Open Execution。"""

    snapshot = build_catalog_snapshot()
    catalog_hash = _write_catalog_package(tmp_path, snapshot)
    frozen = load_frozen_minichem_catalog(task_root=tmp_path, catalog_hash=catalog_hash)

    assert frozen.snapshot["open_execution"] == {
        "schema_version": 1,
        "native_software": [],
        "analysis_runtimes": [],
        "authoring_contract": {
            "path_prefix": "code/",
            "max_files": 4,
            "max_chars_per_file": 100000,
        },
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("unknown", "value", "顶层字段"),
        ("duplicate_module", "numpy", "不能重复"),
        ("absolute_required_file", _ABSOLUTE_INPUT, "绝对路径"),
        ("absolute_argument", _ABSOLUTE_INPUT, "绝对路径"),
    ],
)
def test_load_rejects_corrupt_open_execution_fields(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    """恢复时严格拒绝未知字段、重复模块和命令模板中的绝对路径。"""

    open_snapshot = _open_execution_snapshot()
    command = open_snapshot["native_software"][0]["commands"][0]
    if field == "unknown":
        open_snapshot["unexpected"] = value
    elif field == "duplicate_module":
        open_snapshot["analysis_runtimes"][0]["modules"].append(value)
    elif field == "absolute_required_file":
        command["required_files"] = [value]
    else:
        command["example_request"]["arguments"] = [value]
    catalog_hash = _write_schema2_package(tmp_path, build_catalog_snapshot(), open_snapshot)

    with pytest.raises(MiniChemCatalogFreezeError, match=message):
        load_frozen_minichem_catalog(task_root=tmp_path, catalog_hash=catalog_hash)


def test_load_rejects_hash_complete_snapshot_with_unapproved_budget(tmp_path: Path) -> None:
    """恢复时即使攻击者重算 hash，也必须拒绝 48/204800 旧预算。"""

    snapshot = build_catalog_snapshot()
    snapshot["evaluation_resource_budget"] = {
        "cpu_cores": 48,
        "memory_mb": 204800,
        "gpu_count": 0,
        "source": "evaluation_policy",
        "agent_controllable": False,
        "scope": "per_task",
    }
    catalog_hash = _write_catalog_package(tmp_path, snapshot)

    with pytest.raises(MiniChemCatalogFreezeError, match="资源政策不一致"):
        load_frozen_minichem_catalog(task_root=tmp_path, catalog_hash=catalog_hash)


def test_frozen_catalog_nested_reads_cannot_change_hash_actions_or_projection(tmp_path: Path) -> None:
    """修改冻结对象读出的嵌套值后，目录身份、Action 和 Planner 投影仍不变。"""

    snapshot = build_catalog_snapshot()
    frozen = FrozenMiniChemCatalog(
        catalog_hash=snapshot["catalog_hash"],
        snapshot=snapshot,
        package_root=tmp_path,
    )
    catalog = MiniChemCatalogSnapshot(
        catalog_hash=frozen.catalog_hash,
        catalog_version="7",
        snapshot=frozen.snapshot,
    )
    original_hash = frozen.catalog_hash
    original_actions = catalog.actions
    original_projection = catalog.planner_projection()

    with pytest.raises(TypeError):
        frozen.snapshot["actions"] = []
    frozen.snapshot["actions"][0]["id"] = "tampered_action"
    frozen.snapshot["actions"][0]["backend_ids"].append("tampered_backend")
    frozen.snapshot["evaluation_resource_budget"]["memory_mb"] = 1

    assert frozen.catalog_hash == original_hash
    assert catalog.actions == original_actions
    assert catalog.planner_projection() == original_projection
    assert frozen.snapshot["actions"][0]["id"] == "convert_structure"
    assert frozen.snapshot["evaluation_resource_budget"]["memory_mb"] == 320000


@pytest.mark.parametrize("legacy_layout", ["action_id", "provider_ids", "backend_id"])
def test_load_rejects_legacy_catalog_field_aliases(
    tmp_path: Path,
    legacy_layout: str,
) -> None:
    """恢复时拒绝旧 action/provider/backend 字段，即使旧快照重新计算了 hash。"""

    snapshot = build_catalog_snapshot()
    if legacy_layout == "action_id":
        action = snapshot["actions"][0]
        action[legacy_layout] = action.pop("id")
    elif legacy_layout == "provider_ids":
        action = snapshot["actions"][0]
        action[legacy_layout] = action.pop("backend_ids")
    else:
        backend = snapshot["backends"][0]
        backend[legacy_layout] = backend.pop("id")

    catalog_hash = _write_catalog_package(tmp_path, snapshot)

    with pytest.raises(MiniChemCatalogFreezeError, match="旧字段"):
        load_frozen_minichem_catalog(task_root=tmp_path, catalog_hash=catalog_hash)
