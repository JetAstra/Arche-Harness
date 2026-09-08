"""Software discovery and exact native invocation guidance for open execution."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from minichem_toolbox.catalog import backend_specs
from minichem_toolbox.mini_profile import MINI_NATIVE_SOFTWARE_IDS, MINI_RUNTIME_IDS
from minichem_toolbox.paths import CONFIG_ROOT, PROJECT_ROOT
from minichem_toolbox.runtime import (
    probe_all_backends,
    resolve_executable,
    runtime_names,
    runtime_python,
    runtime_spec,
)
from minichem_toolbox.resource_budget import (
    gaussian_resource_policy_record,
)
from minichem_toolbox.search_index import BM25Index, normalize_scores, weighted_text
from minichem_toolbox.semantic_embeddings import embedding_cache_path, semantic_scores

from .execution_models import (
    AnalysisRuntimeListRequest,
    DocumentationReadRequest,
    DocumentationSearchRequest,
    SoftwareInspectRequest,
    SoftwareListRequest,
)


GUIDE_PATH = CONFIG_ROOT / "native_software_guides.yaml"
EXAMPLE_CONTRACTS_PATH = CONFIG_ROOT / "native_software_example_contracts.yaml"
NATIVE_DOCS_ROOT = PROJECT_ROOT / "docs" / "software"
CAPABILITY_SOURCES_PATH = CONFIG_ROOT / "software_capability_sources.yaml"
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

_PUBLIC_BUDGET = {
    "source": "service_policy",
    "agent_controllable": False,
    "scope": "per_task",
}
_PUBLIC_TIMEOUT = {
    "execution_class": "compute",
    "source": "service_policy",
    "agent_controllable": False,
}
_PUBLIC_HEALTH_FIELDS = frozenset(
    {
        "status",
        "available",
        "reason_code",
        "resource_id",
        "runtime",
        "python_available",
    }
)
_PUBLIC_COMMAND_FIELDS = (
    "executable",
    "enabled",
    "available",
    "synopsis",
    "input_mode",
    "required_files",
    "output_behavior",
    "configuration_contract",
)
_PUBLIC_SOFTWARE_ENTRY_FIELDS = (
    "software_id",
    "display_name",
    "inventory_status",
    "available",
    "available_for_submission",
    "native_available_for_submission",
    "executable_resolved",
    "analysis_runtime_available",
    "interface_smoke_status",
    "scientific_smoke_status",
    "known_runtime_blockers",
    "smoke_evidence",
    "backend_registered",
    "runtime",
    "actions",
    "aliases",
)

_PUBLIC_DOCUMENT_HEADINGS = {
    "Installed software": "Software availability",
}


def _public_document_heading(value: Any) -> str:
    """按标题白名单返回调用方可读的文档导航名称。"""

    heading = str(value)
    return _PUBLIC_DOCUMENT_HEADINGS.get(heading, heading)


def _public_health(value: Any) -> dict[str, Any] | None:
    """只公开软件可用性结论，不回显安装配置和宿主探针明细。"""

    if not isinstance(value, dict):
        return None
    projected: dict[str, Any] = {}
    for name in _PUBLIC_HEALTH_FIELDS:
        if name not in value:
            continue
        item = value[name]
        if name in {"available", "python_available"}:
            projected[name] = bool(item)
        elif name in {"status", "reason_code", "resource_id", "runtime"}:
            projected[name] = str(item)
    return projected


def _public_command(command: dict[str, Any]) -> dict[str, Any]:
    """按命令能力白名单公开调用说明，不公开解析后的可执行文件路径。"""

    projected: dict[str, Any] = {}
    if "executable" in command:
        projected["executable"] = str(command["executable"])
    for key in ("enabled", "available"):
        if key in command:
            projected[key] = bool(command[key])
    for key in ("synopsis", "input_mode", "output_behavior"):
        if key in command and command[key] is not None:
            projected[key] = str(command[key])
    if "required_files" in command:
        projected["required_files"] = [
            str(item) for item in command.get("required_files") or []
        ]
    configuration = command.get("configuration_contract")
    if isinstance(configuration, dict):
        allowed = (
            "tested_version",
            "file_resolution",
            "valid_top_level_sections",
            "endpoint_rule",
        )
        projected_configuration = {
            key: configuration[key]
            for key in allowed
            if key in configuration
        }
        if projected_configuration:
            projected["configuration_contract"] = projected_configuration
    return projected


def _public_inventory_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """按软件清单白名单公开能力状态和 Action 覆盖范围。"""

    projected: dict[str, Any] = {}
    for key in _PUBLIC_SOFTWARE_ENTRY_FIELDS:
        if key not in entry:
            continue
        value = entry[key]
        if key in {
            "available",
            "available_for_submission",
            "native_available_for_submission",
            "executable_resolved",
            "analysis_runtime_available",
            "backend_registered",
        }:
            projected[key] = bool(value)
        elif key in {"actions", "aliases"}:
            projected[key] = [str(item) for item in value or []]
        elif value is not None:
            projected[key] = str(value) if key in {"software_id", "display_name", "runtime"} else value
    projected["native_commands"] = [
        _public_command(command)
        for command in entry.get("native_commands") or []
        if isinstance(command, dict)
    ]
    projected["analysis_runtimes"] = [
        {
            key: runtime.get(key)
            for key in ("runtime", "module", "available", "version")
            if key in runtime
        }
        for runtime in entry.get("analysis_runtimes") or []
        if isinstance(runtime, dict)
    ]
    return projected


@lru_cache(maxsize=1)
def load_native_guides() -> dict[str, Any]:
    value = yaml.safe_load(GUIDE_PATH.read_text(encoding="utf-8")) or {}
    if value.get("schema_version") != 1 or not isinstance(value.get("software"), dict):
        raise ValueError(f"Invalid native software guide: {GUIDE_PATH}")
    return value


@lru_cache(maxsize=1)
def load_native_example_contracts() -> dict[str, Any]:
    value = yaml.safe_load(EXAMPLE_CONTRACTS_PATH.read_text(encoding="utf-8")) or {}
    if value.get("schema_version") != 1 or not isinstance(value.get("software"), dict):
        raise ValueError(f"Invalid native software example contracts: {EXAMPLE_CONTRACTS_PATH}")
    return value


def _smoke_summary(_software_id: str) -> dict[str, Any]:
    return {
        "interface_smoke_status": "not_tested",
        "scientific_smoke_status": "not_tested",
        "known_runtime_blockers": [],
        "smoke_evidence": None,
    }


def software_documentation_recovery(
    software_id: str, *, failed: bool = False
) -> list[dict[str, Any]]:
    sections = [
        ("quickstart", "Toolbox submission request"),
        ("troubleshooting", "Pre-submission checklist"),
    ]
    if failed:
        sections.insert(0, ("troubleshooting", "Known failures and repairs"))
    return [
        {
            "tool": "read_software_documentation",
            "request": {
                "software_id": software_id,
                "topic": topic,
                "section": section,
                "max_chars": 6000,
            },
        }
        for topic, section in sections
    ]


@lru_cache(maxsize=1)
def _source_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    if not CAPABILITY_SOURCES_PATH.is_file():
        return aliases
    value = yaml.safe_load(CAPABILITY_SOURCES_PATH.read_text(encoding="utf-8")) or {}
    for software_id, item in dict(value.get("software") or {}).items():
        aliases[_normalize_id(software_id)] = str(software_id)
        for alias in item.get("aliases") or []:
            aliases[_normalize_id(str(alias))] = str(software_id)
    return aliases


def _normalize_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _native_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for software_id, item in load_native_guides()["software"].items():
        if software_id not in MINI_NATIVE_SOFTWARE_IDS:
            continue
        for alias in item.get("aliases") or []:
            normalized = _normalize_id(str(alias))
            if normalized in aliases and aliases[normalized] != software_id:
                raise ValueError(f"Duplicate native software alias {alias!r}")
            aliases[normalized] = software_id
    return aliases


def _driver_contract(software_id: str) -> dict[str, Any]:
    backend = backend_specs().get(software_id)
    if backend is None:
        raise ValueError(f"Native software {software_id!r} has no BackendSpec")
    backend_commands = set(backend.executables)
    return {
        "software_id": software_id,
        "backend": backend,
        "runtime": backend.runtime,
        "allowed_commands": backend_commands,
        "backend_commands": backend_commands,
    }


def _resolve_guided_executable(
    runtime: str,
    executable: str,
    command_guide: dict[str, Any],
) -> str | None:
    configured = command_guide.get("configured_path")
    if configured:
        path = Path(str(configured)).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        resolved = path.resolve(strict=False)
        try:
            resolved.relative_to(PROJECT_ROOT.resolve())
        except ValueError as exc:
            raise ValueError(
                f"Configured native executable escapes project root: {configured}"
            ) from exc
        return str(resolved) if resolved.is_file() and resolved.stat().st_mode & 0o111 else None
    return resolve_executable(runtime, executable)


def resolve_software_id(value: str) -> str:
    normalized = _normalize_id(value)
    native_aliases = _native_aliases()
    if normalized in native_aliases:
        return native_aliases[normalized]
    known = set(backend_specs()) | set(MINI_NATIVE_SOFTWARE_IDS)
    if normalized in known:
        return normalized
    aliases = _source_aliases()
    if normalized in aliases and aliases[normalized] in known:
        return aliases[normalized]
    for backend_id, backend in backend_specs().items():
        if _normalize_id(backend.display_name) == normalized:
            return backend_id
    raise KeyError(f"Unknown software_id {value!r}")


def validate_native_guides() -> None:
    """Ensure every native command is tied to the focused BackendSpec catalog."""

    specifications = backend_specs()
    covered_backend_commands: dict[str, set[str]] = {}
    for software_id, item in load_native_guides()["software"].items():
        if software_id not in MINI_NATIVE_SOFTWARE_IDS:
            continue
        contract = _driver_contract(software_id)
        commands = item.get("commands") or {}
        if not isinstance(commands, dict) or not commands:
            raise ValueError(f"Native guide {software_id!r} has no commands")
        declared = contract["allowed_commands"]
        unknown = set(commands) - declared
        if unknown:
            raise ValueError(
                f"Native guide {software_id!r} declares commands absent from BackendSpec and "
                f"the focused backend catalog: {sorted(unknown)}"
            )
        backend = contract["backend"]
        for command in commands:
            if contract["runtime"] != backend.runtime:
                raise ValueError(
                    f"Native guide {software_id}/{command} must use BackendSpec runtime "
                    f"{backend.runtime!r}, not {contract['runtime']!r}"
                )
        covered_backend_commands.setdefault(software_id, set()).update(
            set(commands) & contract["backend_commands"]
        )
        for command, guide in commands.items():
            if not str(guide.get("synopsis") or "").strip():
                raise ValueError(f"Native guide {software_id}/{command} lacks synopsis")
    for software_id, specification in specifications.items():
        if specification.executables:
            missing = set(specification.executables) - covered_backend_commands.get(
                software_id, set()
            )
            if missing:
                raise ValueError(
                    f"BackendSpec {software_id!r} lacks invocation guides: {sorted(missing)}"
                )


def native_command_guide(software_id: str, executable: str) -> dict[str, Any]:
    resolved_id = resolve_software_id(software_id)
    if resolved_id not in MINI_NATIVE_SOFTWARE_IDS:
        raise ValueError(f"software_id={resolved_id!r} has no MiniChem native interface")
    contract = _driver_contract(resolved_id)
    backend = contract["backend"]
    if executable not in contract["allowed_commands"]:
        raise ValueError(
            f"{executable!r} is not an executable declared for software_id={resolved_id!r}"
        )
    software = load_native_guides()["software"].get(resolved_id) or {}
    guide = dict((software.get("commands") or {}).get(executable) or {})
    if not guide:
        raise ValueError(
            f"No reviewed native invocation guide exists for {resolved_id}/{executable}"
        )
    if guide.get("enabled", True) is not True:
        raise ValueError(
            f"Native command {resolved_id}/{executable} is intentionally not exposed: "
            f"{guide.get('output_behavior', 'disabled by execution policy')}"
        )
    return {
        "software_id": resolved_id,
        "display_name": backend.display_name,
        "runtime": contract["runtime"],
        "executable": executable,
        "resolved_path": _resolve_guided_executable(
            contract["runtime"], executable, guide
        ),
        **guide,
    }


def _analysis_runtimes_for(
    software_id: str,
    backend: Any | None,
) -> list[dict[str, Any]]:
    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    if backend is not None and runtime_python(backend.runtime).is_file():
        for module in backend.python_modules:
            candidates.setdefault(
                (backend.runtime, module),
                {
                    "runtime": backend.runtime,
                    "module": module,
                    "available": True,
                    "version": None,
                },
            )
    guide_runtime = (
        (load_native_guides()["software"].get(software_id) or {}).get("runtime")
    )
    if guide_runtime and runtime_python(str(guide_runtime)).is_file():
        candidates.setdefault(
            (str(guide_runtime), "<caller-supplied program>"),
            {
                "runtime": str(guide_runtime),
                "module": "<caller-supplied program>",
                "available": True,
                "version": None,
            },
        )
    return sorted(candidates.values(), key=lambda item: (item["runtime"], item["module"]))


def _native_commands(software_id: str, backend: Any | None) -> list[dict[str, Any]]:
    guide = load_native_guides()["software"].get(software_id) or {}
    if not guide:
        return []
    contract = _driver_contract(software_id)
    result = []
    for executable, command in dict(guide.get("commands") or {}).items():
        enabled = command.get("enabled", True) is True
        resolved = (
            _resolve_guided_executable(contract["runtime"], executable, command)
            if enabled
            else None
        )
        result.append(
            {
                "executable": executable,
                "enabled": enabled,
                "available": bool(resolved),
                "resolved_path": resolved,
                "synopsis": command.get("synopsis"),
                "input_mode": command.get("input_mode"),
            }
        )
    return result


def _inventory_entries() -> list[dict[str, Any]]:
    specs = backend_specs()
    ids = sorted(set(specs) | set(MINI_NATIVE_SOFTWARE_IDS))
    entries: list[dict[str, Any]] = []
    for software_id in ids:
        backend = specs.get(software_id)
        native_commands = _native_commands(software_id, backend)
        analysis_runtimes = _analysis_runtimes_for(software_id, backend)
        smoke = _smoke_summary(software_id)
        executable_resolved = any(item["available"] for item in native_commands)
        native_available = executable_resolved and smoke["interface_smoke_status"] not in {
            "failed",
            "skipped",
        }
        analysis_available = any(
            item["available"] and item["module"] != "<caller-supplied program>"
            for item in analysis_runtimes
        )
        if backend is not None and not backend.executables and not backend.python_modules:
            health = probe_all_backends((backend,)).get(software_id, {})
            analysis_available = bool(health.get("available"))
        available = native_available or analysis_available
        entries.append(
            {
                "software_id": software_id,
                "display_name": backend.display_name if backend is not None else software_id,
                "inventory_status": "configured" if available else "unavailable",
                "available": available,
                "available_for_submission": available,
                "native_available_for_submission": native_available,
                "executable_resolved": executable_resolved,
                "analysis_runtime_available": analysis_available,
                **smoke,
                "backend_registered": backend is not None,
                "runtime": (
                    _driver_contract(software_id)["runtime"]
                    if software_id in load_native_guides()["software"]
                    else (backend.runtime if backend is not None else None)
                ),
                "actions": sorted(backend.capabilities if backend is not None else ()),
                "native_commands": native_commands,
                "analysis_runtimes": analysis_runtimes,
                "aliases": sorted(
                    alias
                    for alias, target in _native_aliases().items()
                    if target == software_id
                ),
            }
        )
    return entries


def list_software(request: SoftwareListRequest) -> dict[str, Any]:
    validate_native_guides()
    entries = _inventory_entries()
    if request.query:
        needle = request.query.casefold()
        entries = [
            item
            for item in entries
            if needle
            in " ".join(
                [
                    item["software_id"],
                    item["display_name"],
                    " ".join(item["actions"]),
                    " ".join(command["executable"] for command in item["native_commands"]),
                ]
            ).casefold()
        ]
    if request.native_only:
        entries = [item for item in entries if item["native_commands"]]
    if request.available_only:
        entries = [item for item in entries if item["available"]]
    selected = entries[request.offset : request.offset + request.limit]
    expose_capabilities = bool(request.query)
    compact = []
    for item in selected:
        native_executables = [
            {
                "executable": command["executable"],
                "enabled": command["enabled"],
                "available": command["available"],
            }
            for command in item["native_commands"]
        ]
        analysis_runtime_ids = sorted(
            {
                runtime["runtime"]
                for runtime in item["analysis_runtimes"]
                if runtime.get("runtime")
            }
        )
        compact.append(
            {
                "software_id": item["software_id"],
                "display_name": item["display_name"],
                "available": item["available"],
                "interface_smoke_status": item["interface_smoke_status"],
                "scientific_smoke_status": item["scientific_smoke_status"],
                "backend_registered": item["backend_registered"],
                "runtime": item["runtime"],
                "native_executables": native_executables,
                "analysis_runtime_ids": analysis_runtime_ids,
                "action_count": len(item["actions"]),
                **({"matching_action_ids": item["actions"]} if expose_capabilities else {}),
            }
        )
    next_offset = request.offset + len(selected)
    return {
        "status": "success",
        "count": len(compact),
        "total_matching": len(entries),
        "offset": request.offset,
        "next_offset": next_offset if next_offset < len(entries) else None,
        "software": compact,
        "evaluation_resource_budget": dict(_PUBLIC_BUDGET),
        "selection_note": (
            "This is a compact inventory filter only. It does not recommend software, rank "
            "backends, or choose a scientific method. Filter with query to expose matching Action "
            "ids, then call inspect_software for exactly one software_id to load command synopses, "
            "documentation topics, and request templates."
        ),
        "pagination_note": (
            "Use next_offset with the same filters to continue until it is null."
        ),
    }


def inspect_software(request: SoftwareInspectRequest) -> dict[str, Any]:
    validate_native_guides()
    software_id = resolve_software_id(request.software_id)
    backend = backend_specs().get(software_id)
    entry = next(item for item in _inventory_entries() if item["software_id"] == software_id)
    health = probe_all_backends((backend,)).get(software_id) if backend is not None else None
    guide = load_native_guides()["software"].get(software_id) or {}
    example_contracts = (
        load_native_example_contracts().get("software", {}).get(software_id, {}).get("commands", {})
    )
    contract = _driver_contract(software_id) if guide else None
    detailed_commands = []
    for executable, command in dict(guide.get("commands") or {}).items():
        example = dict(example_contracts.get(executable) or {})
        enabled = command.get("enabled", True) is True
        resolved = (
            _resolve_guided_executable(contract["runtime"], executable, command)
            if enabled and contract is not None
            else None
        )
        native_template = {
            "software_id": software_id,
            "executable": executable,
            "arguments": [],
            "staged_inputs": [
                {
                    "source_path": f"code/{target}",
                    "target_path": target,
                }
                for target in example.get("inputs") or []
            ],
            "stdin_target": example.get("stdin_target"),
        }
        if software_id == "gaussian":
            native_template["resource_policy"] = gaussian_resource_policy_record()
        # 普通程序的 ResourceLimits 由请求模型的服务端默认值处理；公共模板不公开实际预算。
        detailed_commands.append(
            {
                **_public_command(
                    {
                        **command,
                        "executable": executable,
                        "enabled": enabled,
                        "available": bool(resolved),
                    }
                ),
                "native_job_request_template": native_template,
            }
        )
    document_chunks = software_document_chunks(software_id, include_cached=False)
    document_index: dict[str, dict[str, Any]] = {}
    for chunk in document_chunks:
        if chunk["shared"]:
            continue
        topic_key = "-".join(chunk["topics"]) or "general"
        item = document_index.setdefault(
            topic_key,
            {
                "topics": chunk["topics"],
                "aliases": chunk["aliases"],
                "sections": [],
            },
        )
        if chunk["heading"] not in item["sections"]:
            item["sections"].append(_public_document_heading(chunk["heading"]))
    public_entry = _public_inventory_entry(entry)
    return {
        "status": "success",
        **public_entry,
        "evaluation_resource_budget": dict(_PUBLIC_BUDGET),
        "execution_timeout_policy": dict(_PUBLIC_TIMEOUT),
        "purpose": "Provides the registered software capability for explicit scientific execution.",
        "backend_health": _public_health(health),
        "native_invocation_guides": detailed_commands,
        "documentation_index": list(document_index.values()),
        "shared_documentation_topics": sorted(
            {
                topic
                for chunk in document_chunks
                if chunk["shared"]
                for topic in chunk["topics"]
            }
        ),
        "recommended_documentation_routes": software_documentation_recovery(
            software_id, failed=bool(entry["known_runtime_blockers"])
        ),
        "decision_boundary": (
            "The guide explains how to invoke this software. The caller remains responsible for "
            "whether to use it, the scientific input content, parameters, call order, and interpretation."
        ),
    }


def _normalize_topic(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def _parse_markdown(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end < 0:
        raise ValueError(f"Unclosed YAML front matter in {path}")
    metadata = yaml.safe_load(text[4:end]) or {}
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid YAML front matter in {path}")
    return metadata, text[end + 5 :]


def _markdown_chunks(path: Path) -> list[dict[str, Any]]:
    metadata, body = _parse_markdown(path)
    headings: list[tuple[str, list[str]]] = []
    current_heading = "Document"
    current_lines: list[str] = []
    for line in body.splitlines():
        match = _HEADING.match(line)
        if match:
            if current_lines or current_heading != "Document":
                headings.append((current_heading, current_lines))
            current_heading = match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)
    headings.append((current_heading, current_lines))
    relative = path.relative_to(PROJECT_ROOT)
    topics = [_normalize_topic(str(item)) for item in metadata.get("topics") or []]
    aliases = [str(item) for item in metadata.get("aliases") or []]
    software_id = str(metadata.get("software_id") or path.parent.name)
    chunks = []
    for index, (heading, lines) in enumerate(headings):
        content = "\n".join(lines).strip()
        if not content and heading == "Document":
            continue
        chunks.append(
            {
                "chunk_id": f"{relative}#{index}-{_normalize_topic(heading)}",
                "software_id": software_id,
                "path": str(relative),
                "heading": heading,
                "section": _normalize_topic(heading),
                "topics": topics,
                "aliases": aliases,
                "text": content,
                "shared": software_id == "_shared",
                "source_type": "first_party_markdown",
            }
        )
    return chunks


@lru_cache(maxsize=64)
def _first_party_chunks(software_id: str) -> tuple[dict[str, Any], ...]:
    paths = sorted((NATIVE_DOCS_ROOT / "_shared").glob("*.md"))
    software_directory = NATIVE_DOCS_ROOT / software_id
    if software_directory.is_dir():
        paths.extend(sorted(software_directory.glob("*.md")))
    return tuple(chunk for path in paths for chunk in _markdown_chunks(path))


def software_document_chunks(
    software_id: str, *, include_cached: bool = True
) -> list[dict[str, Any]]:
    resolved_id = resolve_software_id(software_id)
    return [dict(item) for item in _first_party_chunks(resolved_id)]


def _topic_matches(chunk: dict[str, Any], value: str) -> bool:
    target = _normalize_topic(value)
    candidates = {
        *chunk["topics"],
        *(_normalize_topic(item) for item in chunk["aliases"]),
    }
    return target in candidates


def software_document_search_text(chunk: dict[str, Any]) -> str:
    return weighted_text(
        (
            (chunk["heading"], 4),
            (" ".join(chunk["topics"]), 4),
            (" ".join(chunk["aliases"]), 3),
            (chunk["text"], 1),
        )
    )


def read_software_documentation(request: DocumentationReadRequest) -> dict[str, Any]:
    software_id = resolve_software_id(request.software_id)
    matches = [
        chunk
        for chunk in software_document_chunks(software_id, include_cached=False)
        if _topic_matches(chunk, request.topic)
    ]
    if request.section:
        section = _normalize_topic(request.section)
        matches = [chunk for chunk in matches if chunk["section"] == section]
    if not matches:
        raise KeyError(
            f"No first-party documentation for software_id={software_id!r}, "
            f"topic={request.topic!r}, section={request.section!r}"
        )
    remaining = request.max_chars
    sections = []
    truncated = False
    for chunk in matches:
        content = chunk["text"][:remaining]
        sections.append(
            {
                "heading": chunk["heading"],
                "content": content,
                "character_count": len(content),
                "word_count": len(content.split()),
                "estimated_context_tokens": (len(content) + 3) // 4,
            }
        )
        remaining -= len(content)
        if remaining <= 0:
            truncated = True
            break
    character_count = sum(item["character_count"] for item in sections)
    return {
        "status": "success",
        "software_id": software_id,
        "topic": _normalize_topic(request.topic),
        "section": _normalize_topic(request.section) if request.section else None,
        "sections": sections,
        "truncated": truncated,
        "source_policy": "registered_documentation_exact_route",
        "retrieval_usage": {
            "character_count": character_count,
            "word_count": sum(item["word_count"] for item in sections),
            "estimated_context_tokens": (character_count + 3) // 4,
            "token_estimate_policy": "UTF-8 character count divided by four; this is an approximate context-size estimate",
        },
    }


def search_software_documentation(request: DocumentationSearchRequest) -> dict[str, Any]:
    software_id = resolve_software_id(request.software_id)
    all_chunks = software_document_chunks(software_id, include_cached=True)
    candidates = list(all_chunks)
    if request.topic:
        candidates = [chunk for chunk in candidates if _topic_matches(chunk, request.topic)]
    if request.section:
        section = _normalize_topic(request.section)
        candidates = [chunk for chunk in candidates if chunk["section"] == section]
    documents = {
        chunk["chunk_id"]: software_document_search_text(chunk)
        for chunk in all_chunks
    }
    semantic_documents = {
        chunk["chunk_id"]: documents[chunk["chunk_id"]]
        for chunk in all_chunks
        if chunk["source_type"] == "first_party_markdown"
    }
    scores: dict[str, dict[str, float]] = {}
    semantic_status = "not_requested"
    if request.query:
        lexical_raw = {
            item.document_id: item.score
            for item in BM25Index(documents).search(request.query)
        }
        lexical = normalize_scores(lexical_raw)
        semantic: dict[str, float] = {}
        if request.retrieval_mode == "hybrid":
            semantic_raw, semantic_status = semantic_scores(
                request.query,
                semantic_documents,
                cache_path=embedding_cache_path().parent
                / f"software_docs_{software_id}.npz",
            )
            semantic = normalize_scores(
                {key: max(0.0, value) for key, value in semantic_raw.items() if value >= 0.20}
            )
        for chunk in candidates:
            chunk_id = chunk["chunk_id"]
            lexical_score = lexical.get(chunk_id, 0.0)
            semantic_score = semantic.get(chunk_id, 0.0)
            if lexical_score or semantic_score:
                scores[chunk_id] = {
                    "combined": lexical_score + 0.25 * semantic_score,
                    "lexical": lexical_score,
                    "semantic": semantic_score,
                }
        candidates = [chunk for chunk in candidates if chunk["chunk_id"] in scores]
        candidates.sort(
            key=lambda chunk: (-scores[chunk["chunk_id"]]["combined"], chunk["chunk_id"])
        )
    else:
        candidates.sort(key=lambda chunk: chunk["chunk_id"])
    results = []
    for chunk in candidates[: request.max_results]:
        text = chunk["text"]
        excerpt = text[: request.context_chars]
        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "heading": chunk["heading"],
                "topics": chunk["topics"],
                "source_type": "registered_documentation",
                "score": scores.get(chunk["chunk_id"]),
                "excerpt": excerpt,
                "character_count": len(excerpt),
                "word_count": len(excerpt.split()),
                "estimated_context_tokens": (len(excerpt) + 3) // 4,
            }
        )
    character_count = sum(item["character_count"] for item in results)
    return {
        "status": "success",
        "software_id": software_id,
        "query": request.query,
        "topic": request.topic,
        "section": request.section,
        "match_count": len(results),
        "matches": results,
        "retrieval": {
            "mode": request.retrieval_mode,
            "exact_route_first": True,
            "lexical_ranker": "lexical",
            "semantic_model": None,
            "semantic_status": (
                "available"
                if semantic_status == "available"
                else "not_requested"
                if semantic_status == "not_requested"
                else "unavailable"
            ),
            "returned_character_count": character_count,
            "returned_word_count": sum(item["word_count"] for item in results),
            "estimated_context_tokens": (character_count + 3) // 4,
            "token_estimate_policy": "UTF-8 character count divided by four; this is an approximate context-size estimate",
        },
        "note": "Registered documentation is indexed by heading.",
    }


def list_analysis_runtimes(request: AnalysisRuntimeListRequest) -> dict[str, Any]:
    result = []
    for name in runtime_names():
        if name not in MINI_RUNTIME_IDS:
            continue
        specification = runtime_spec(name)
        python = runtime_python(name)
        available = python.is_file()
        if request.available_only and not available:
            continue
        if request.runtime is not None and name != request.runtime:
            continue
        modules = {
            module: {"available": None, "version": None}
            for module in (
                (specification.get("health_checks") or {}).get("modules")
                or specification.get("modules")
                or []
            )
        }
        commands = sorted(
            set(
                (specification.get("health_checks") or {}).get("commands")
                or specification.get("commands")
                or []
            )
        )
        backends = specification.get("backends") or []
        query_haystack = " ".join(
            [
                name,
                str(specification.get("description") or ""),
                *modules,
                *commands,
                *backends,
            ]
        ).casefold()
        if request.query and request.query.casefold() not in query_haystack:
            continue
        item: dict[str, Any] = {
            "runtime": name,
            "description": "Registered chemistry runtime for caller-supplied analysis programs.",
            "available": available,
            "module_count": len(modules),
            "configured_command_count": len(commands),
            "backend_ids": backends,
        }
        if request.query or request.runtime is not None:
            item["module_names"] = sorted(modules)
        if request.include_details:
            item.update(
                {
                    "python_available": available,
                    "modules": modules,
                }
            )
        result.append(item)
    total_matches = len(result)
    result = result[: request.limit]
    return {
        "status": "success",
        "managed_program_contract": {
            "job_context_required_for_compliance": True,
            "execution_note": (
                "Submit the caller-supplied program with submit_analysis_program. Declare named "
                "inputs and outputs, and use JobContext helpers whose names match those declarations."
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
                "JobContext helper names must exactly match the corresponding input/output "
                "declaration names."
            ),
        },
        "submit_analysis_program_request_template": {
            "runtime": request.runtime or "<exact runtime id>",
            "script_path": "code/<user-program>.py",
            "script_target": "code/user_program.py",
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
        },
        "count": len(result),
        "total_matches": total_matches,
        "runtimes": result,
        "evaluation_resource_budget": dict(_PUBLIC_BUDGET),
        "execution_timeout_policy": dict(_PUBLIC_TIMEOUT),
        "selection_note": (
            "Select one runtime explicitly according to the imports required by the caller-supplied "
            "program. No runtime is chosen automatically. Use runtime=<exact id> and "
            "include_details=true after narrowing to inspect declared modules."
        ),
        "execution_note": (
            "Submit a caller-supplied scientific program with submit_analysis_program. Declare "
            "inputs and outputs, select one runtime, and use JobContext for declared data."
        ),
    }


def open_execution_prompt(*, include_command_index: bool = True) -> str:
    """Return a neutral description of layers two and three.

    Progressive discovery omits the eager command synopsis index because the
    same exact information is available on demand through ``list_software`` and
    ``inspect_software``.
    """

    lines = [
        "",
        "### Layer 2: software-native execution",
        "When no predefined Action expresses the required operation, you may write native input "
        "files and invoke an allowlisted command. First call `inspect_software` for the exact "
        "software id, executable, input mode, required filenames, synopsis, and request template. "
        "Then use `write_workspace_text`, `validate_native_job`, and `submit_native_job`. The "
        "runner uses no shell and does not select software or add scientific parameters. Poll with "
        "`get_execution_job` and collect declared outputs with `collect_execution_job`. At any "
        "point, call `list_execution_jobs` to recover every submitted or historical job in the "
        "current capability scope and inspect all active_job_ids before polling or collecting.",
        "",
    ]
    if include_command_index:
        lines.append("Registered native command ids and invocation synopses:")
        for software_id, item in load_native_guides()["software"].items():
            if software_id not in MINI_NATIVE_SOFTWARE_IDS:
                continue
            commands = []
            for executable, command in (item.get("commands") or {}).items():
                if command.get("enabled", True) is True:
                    commands.append(f"{executable}: {str(command.get('synopsis') or '')}")
            if commands:
                lines.append(f"- `{software_id}` - " + " | ".join(commands))
    else:
        lines.append(
            "The complete software inventory and every command synopsis remain available "
            "on demand through `list_software` and `inspect_software`; no program is hidden or "
            "selected automatically."
        )
    lines.extend(
        [
            "",
            "### Layer 3: programmable scientific analysis",
            "For an operation that is best expressed by code, call `list_analysis_runtimes`, write a "
            "complete Python file under code/, and submit it with `submit_analysis_program` in one "
            "explicit runtime. Declare inputs and outputs and read declared inputs with "
            "`JobContext.input`. The program and staged inputs are recorded with the job; stdout, "
            "stderr, exit status, and output hashes use the same job record as native software. "
            "Use `declare_scientific_artifact` to attach semantic type and parent lineage to important "
            "outputs. The programmable layer is resource-confined but is not a replacement for an "
            "OS/container security boundary.",
            "",
            "The three layers are peers, not a hidden workflow. A predefined Action is a convenient "
            "validated operation when it matches. Native execution and caller-supplied programs are "
            "available for capabilities that were not anticipated by the Action catalog. You decide "
            "the complete plan and may interleave all three layers.",
        ]
    )
    return "\n".join(lines)


def software_resource_snapshot() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "execution_policy": {
            "shell": False,
            "automatic_scientific_defaults": False,
            "automatic_backend_selection": False,
            "automatic_fallback": False,
            "request_requirements": [
                "software_id",
                "executable",
                "arguments",
                "staged_inputs",
            ],
        },
        **list_software(SoftwareListRequest(limit=200)),
    }


__all__ = [
    "inspect_software",
    "list_analysis_runtimes",
    "list_software",
    "load_native_guides",
    "native_command_guide",
    "open_execution_prompt",
    "read_software_documentation",
    "resolve_software_id",
    "search_software_documentation",
    "software_document_chunks",
    "software_document_search_text",
    "software_resource_snapshot",
    "validate_native_guides",
]
