"""FastMCP bindings for the software-native and programmable execution layers."""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from pydantic import BaseModel

from .execution_models import (
    AnalysisInputInspectionRequest,
    AnalysisJobRequest,
    AnalysisRuntimeListRequest,
    ArtifactDeclarationRequest,
    DocumentationReadRequest,
    DocumentationSearchRequest,
    ExecutionJobListRequest,
    ExecutionResourceRequest,
    JobCancelRequest,
    JobCollectRequest,
    JobStatusRequest,
    NativeJobRequest,
    ScientificArtifactTextReadRequest,
    SoftwareInspectRequest,
    SoftwareListRequest,
    VerifyExecutionReferencesRequest,
    WorkspaceTextReadRequest,
    WorkspaceTextWriteRequest,
)
from .open_execution import (
    cancel_execution_job as _cancel_execution_job,
    collect_execution_job as _collect_execution_job,
    declare_scientific_artifact as _declare_scientific_artifact,
    get_execution_job as _get_execution_job,
    get_execution_resources as _get_execution_resources,
    inspect_analysis_inputs as _inspect_analysis_inputs,
    list_execution_jobs as _list_execution_jobs,
    read_scientific_artifact_text as _read_scientific_artifact_text,
    read_workspace_text as _read_workspace_text,
    submit_analysis_program as _submit_analysis_program,
    submit_native_job as _submit_native_job,
    validate_native_job as _validate_native_job,
    validate_analysis_program as _validate_analysis_program,
    verify_execution_references as _verify_execution_references,
    write_workspace_text as _write_workspace_text,
)
from .software_catalog import (
    inspect_software as _inspect_software,
    list_analysis_runtimes as _list_analysis_runtimes,
    list_software as _list_software,
    read_software_documentation as _read_software_documentation,
    search_software_documentation as _search_software_documentation,
    software_documentation_recovery,
)
from .tracing import execute_traced


RequestT = TypeVar("RequestT", bound=BaseModel)


OPEN_EXECUTION_TOOL_NAMES = (
    "list_software",
    "inspect_software",
    "read_software_documentation",
    "search_software_documentation",
    "write_workspace_text",
    "read_workspace_text",
    "validate_native_job",
    "submit_native_job",
    "list_analysis_runtimes",
    "inspect_analysis_inputs",
    "validate_analysis_program",
    "submit_analysis_program",
    "get_execution_resources",
    "list_execution_jobs",
    "get_execution_job",
    "collect_execution_job",
    "cancel_execution_job",
    "declare_scientific_artifact",
)
ARTIFACT_CAPTURING_TOOLS = {
    "write_workspace_text",
    "declare_scientific_artifact",
}


TOOL_DESCRIPTIONS = {
    "list_software": (
        "List and filter a paginated index of registered software capabilities. Use query to narrow "
        "it, then inspect_software for one exact id and its command/input/output contract. This "
        "operation does not recommend, rank, or select a program."
    ),
    "inspect_software": (
        "Inspect one exact software id before using it. Returns its registered capability, command "
        "synopsis, input mode, required filenames, output behavior, documentation topics, and a "
        "request template. Use this to construct the command request."
    ),
    "read_software_documentation": (
        "Read a bounded first-party Markdown topic for one exact software id, optionally narrowed "
        "to one heading. Use inspect_software to discover topic ids."
    ),
    "search_software_documentation": (
        "Search heading-level registered documentation for one exact software id. Exact topic and "
        "section routing is applied first, followed by lexical relevance and optional semantic recall."
    ),
    "write_workspace_text": (
        "Write a caller-supplied UTF-8 input deck, configuration, or Python program under code/ or "
        "outputs/. Existing files are protected unless overwrite=true is explicit. Framework control "
        "state is not a writable workspace target."
    ),
    "read_workspace_text": (
        "Read an exact user input, result, report, or output UTF-8 workspace file with a bounded "
        "character count and content hash. Framework control state is not readable through this "
        "tool. "
        "Use offset from the previous response's next_offset to continue reading; the response "
        "reports offset, returned_chars, total_chars, truncated, and next_offset (null at EOF). "
        "For UTF-8 JSON files, optionally use an RFC 6901 json_pointer ('' selects the root) "
        "to read one exact object, array item, or scalar before applying offset/max_chars."
    ),
    "validate_native_job": (
        "Validate one exact caller-supplied native invocation without executing it. Checks the "
        "software/executable allowlist, availability, argv safety, staging/stdin contract, "
        "and mechanical resources. Gaussian CPU and memory use service_managed_full_capacity with "
        "serial execution; caller-supplied values are not controlling inputs. It never adds "
        "scientific parameters or judges the method."
    ),
    "submit_native_job": (
        "Submit one exact reviewed software command asynchronously with shell=false. All input files "
        "must be mapped explicitly into an isolated job directory. Returns a persistent job_id; no "
        "software choice, scientific default, callback, retry, or fallback is performed. Gaussian "
        "jobs use service_managed_full_capacity and run serially; caller CPU or memory values do not "
        "change the job."
    ),
    "list_analysis_runtimes": (
        "Search the configured Python runtimes available for a caller-supplied scientific program. "
        "The default response is a compact inventory; filter by query/runtime and request details "
        "only for the selected environment. No runtime is selected automatically."
    ),
    "inspect_analysis_inputs": (
        "Inspect bounded structural metadata for declared analysis inputs before writing or "
        "submitting a program. JSON keys/types/list lengths/nulls and CSV/TSV columns/types are "
        "reported without selecting a scientific method or interpreting the result."
    ),
    "submit_analysis_program": (
        "Submit a caller-supplied .py file asynchronously in one explicitly selected chemistry "
        "runtime. The script and inputs are staged and hashed; stdout/stderr/resources/outputs share "
        "the native job record. This process confinement is not an OS container security boundary."
    ),
    "validate_analysis_program": (
        "Preflight a caller-supplied Python analysis without executing it. Checks UTF-8 syntax, "
        "imports in the selected runtime, declared inputs and outputs, paths, and resources, then "
        "returns the exact job layout and structured repair diagnostics."
    ),
    "get_execution_resources": (
        "Read the service-managed resource availability state before choosing submission concurrency. "
        "The response also contains a shared Gaussian FIFO view ordered by ticket. Within an "
        "authenticated scope, your own entries include the exact job_id and owned=true; entries "
        "owned by another scope contain only position, status, and owned=false. No anonymous job "
        "reference, PID, path, or cross-scope job identity is exposed."
    ),
    "list_execution_jobs": (
        "List every submitted or historical execution job in the current capability scope. "
        "The compact, stable response includes job_id, persistent job_status, registry phase, "
        "terminal, and safe submission_key/operation_id identity when available, plus a stable "
        "active_job_ids list. For backward compatibility, active_job_id is populated only when "
        "there is exactly one active job and is null when there are none or several. Use each "
        "returned exact job_id with get_execution_job to poll and collect_execution_job to retrieve "
        "terminal results. It never exposes paths, PIDs, logs, or jobs from another scope."
    ),
    "get_execution_job": (
        "Poll one submitted execution job by exact job_id. The unified response exposes only the "
        "common job_id, job_status, terminal state, and a minimal job envelope; it does not expose "
        "job type, private metadata, or logs."
    ),
    "collect_execution_job": (
        "Collect one exact terminal job by job_id and its matching submission_key. The service "
        "routes the private result materialization to the job handler, preserving TS canonical "
        "ActionResult/artifact checks and the generic output manifest without interpreting science."
    ),
    "cancel_execution_job": (
        "Send a cancellation request to one exact running job. It never starts a replacement or "
        "changes scientific parameters."
    ),
    "declare_scientific_artifact": (
        "Register an existing workspace file as an immutable ArtifactRef with caller-supplied semantic "
        "type, producer layer/id, media type, and parent lineage. No semantics are inferred."
    ),
    "read_scientific_artifact_text": (
        "Read a bounded UTF-8 text ArtifactRef by artifact_id in the current capability scope. "
        "The canonical manifest and actual SHA-256 are checked on every page; the response "
        "contains no host path, log, or PID."
    ),
    "verify_execution_references": (
        "Read-only verification of explicitly supplied job, typed Action operation, and ArtifactRef "
        "identities in the current capability scope. Checks terminal job state, operation fingerprints, "
        "canonical artifact manifests, and actual SHA-256 bytes without re-executing or returning host "
        "paths, logs, PIDs, or stored Action responses. The response also reports the complete typed "
        "operation inventory and its active state; explicit operation refs must cover that inventory. "
        "Unknown or cross-scope references are unavailable."
    ),
}


def _invoke(
    name: str,
    request: RequestT,
    function: Callable[[RequestT], dict[str, Any]],
) -> dict[str, Any]:
    arguments = {"request": request.model_dump(mode="json")}

    def run() -> dict[str, Any]:
        try:
            return function(request)
        except (FileNotFoundError, FileExistsError, KeyError, UnicodeError, ValueError) as exc:
            result = {
                "status": "invalid_request",
                "error": {
                    "code": "invalid_open_execution_request",
                    "message": str(exc),
                    "exception_type": type(exc).__name__,
                },
                "automatic_fallback": False,
            }
            if name in {"validate_native_job", "submit_native_job"} and isinstance(
                request, NativeJobRequest
            ):
                result["documentation_recovery"] = software_documentation_recovery(
                    request.software_id, failed=True
                )
            return result
        except OSError as exc:
            return {
                "status": "failed",
                "error": {
                    "code": "open_execution_os_error",
                    "message": str(exc),
                    "exception_type": type(exc).__name__,
                },
                "automatic_fallback": False,
            }
        except Exception as exc:
            return {
                "status": "failed",
                "error": {
                    "code": "open_execution_internal_error",
                    "message": str(exc),
                    "exception_type": type(exc).__name__,
                },
                "automatic_fallback": False,
            }

    return execute_traced(
        name,
        arguments,
        run,
        capture_artifacts=name in ARTIFACT_CAPTURING_TOOLS,
    )


def list_software(request: SoftwareListRequest) -> dict[str, Any]:
    return _invoke("list_software", request, _list_software)


def inspect_software(request: SoftwareInspectRequest) -> dict[str, Any]:
    return _invoke("inspect_software", request, _inspect_software)


def read_software_documentation(request: DocumentationReadRequest) -> dict[str, Any]:
    return _invoke(
        "read_software_documentation", request, _read_software_documentation
    )


def search_software_documentation(request: DocumentationSearchRequest) -> dict[str, Any]:
    return _invoke(
        "search_software_documentation", request, _search_software_documentation
    )


def write_workspace_text(request: WorkspaceTextWriteRequest) -> dict[str, Any]:
    return _invoke("write_workspace_text", request, _write_workspace_text)


def read_workspace_text(request: WorkspaceTextReadRequest) -> dict[str, Any]:
    return _invoke("read_workspace_text", request, _read_workspace_text)


def validate_native_job(request: NativeJobRequest) -> dict[str, Any]:
    return _invoke("validate_native_job", request, _validate_native_job)


def submit_native_job(request: NativeJobRequest) -> dict[str, Any]:
    return _invoke("submit_native_job", request, _submit_native_job)


def list_analysis_runtimes(request: AnalysisRuntimeListRequest) -> dict[str, Any]:
    return _invoke("list_analysis_runtimes", request, _list_analysis_runtimes)


def inspect_analysis_inputs(request: AnalysisInputInspectionRequest) -> dict[str, Any]:
    return _invoke("inspect_analysis_inputs", request, _inspect_analysis_inputs)


def validate_analysis_program(request: AnalysisJobRequest) -> dict[str, Any]:
    return _invoke("validate_analysis_program", request, _validate_analysis_program)


def submit_analysis_program(request: AnalysisJobRequest) -> dict[str, Any]:
    return _invoke("submit_analysis_program", request, _submit_analysis_program)


def get_execution_resources(request: ExecutionResourceRequest) -> dict[str, Any]:
    return _invoke("get_execution_resources", request, _get_execution_resources)


def list_execution_jobs(request: ExecutionJobListRequest) -> dict[str, Any]:
    return _invoke("list_execution_jobs", request, _list_execution_jobs)


def get_execution_job(request: JobStatusRequest) -> dict[str, Any]:
    return _invoke("get_execution_job", request, _get_execution_job)


def collect_execution_job(request: JobCollectRequest) -> dict[str, Any]:
    return _invoke("collect_execution_job", request, _collect_execution_job)


def cancel_execution_job(request: JobCancelRequest) -> dict[str, Any]:
    return _invoke("cancel_execution_job", request, _cancel_execution_job)


def declare_scientific_artifact(request: ArtifactDeclarationRequest) -> dict[str, Any]:
    return _invoke(
        "declare_scientific_artifact", request, _declare_scientific_artifact
    )


def read_scientific_artifact_text(
    request: ScientificArtifactTextReadRequest,
) -> dict[str, Any]:
    return _invoke(
        "read_scientific_artifact_text", request, _read_scientific_artifact_text
    )


def verify_execution_references(
    request: VerifyExecutionReferencesRequest,
) -> dict[str, Any]:
    return _invoke(
        "verify_execution_references", request, _verify_execution_references
    )


def register_open_execution_tools(mcp) -> list[str]:
    functions = {
        name: globals()[name]
        for name in OPEN_EXECUTION_TOOL_NAMES
    }
    for name, function in functions.items():
        mcp.tool(name=name, description=TOOL_DESCRIPTIONS[name])(function)
    return list(functions)


__all__ = [
    *OPEN_EXECUTION_TOOL_NAMES,
    "OPEN_EXECUTION_TOOL_NAMES",
    "register_open_execution_tools",
    "read_scientific_artifact_text",
    "verify_execution_references",
]
