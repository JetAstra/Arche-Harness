import argparse
import json
import os
import time
import importlib
from typing import Any, Dict, List, Optional

import openai


DEFAULT_MODEL_CONFIG = {
    "temperature": 0.7,
    "max_tokens": 1024,
    "model": (
        os.getenv("ARCHE_CHEM_MODEL_PATH")
        or os.getenv("GAUSSIAN_LOCAL_MODEL_PATH")
        or ""
    ),
    "model_name": (
        os.getenv("ARCHE_CHEM_MODEL_NAME")
        or os.getenv("GAUSSIAN_LOCAL_MODEL_NAME")
        or "deepseek-chat"
    ),
    "backend": (
        os.getenv("ARCHE_CHEM_BACKEND")
        or os.getenv("GAUSSIAN_LOCAL_BACKEND")
        or "vllm"
    ),
    "parallel_size": 1,
    "gpu_memory_utilization": 0.7,
}
_LOCAL_LLM_CACHE: Dict[str, Any] = {}
_VLLM_RUNTIME: Dict[str, Any] = {}
_ARCHE_CHEM_CALL_FN: Optional[Any] = None
_ARCHE_CHEM_CALL_LOADED = False
DEFAULT_TIMEOUT = int(os.getenv("GAUSSIAN_CODE_TIMEOUT", "120"))
DEFAULT_API_BASE_URL = os.getenv("ARCHE_CHEM_BASE_URL", "https://api.deepseek.com/v1")
DEFAULT_API_MODEL = os.getenv("ARCHE_CHEM_MODEL_NAME", "deepseek-v4-pro")


class LocalBackendUnavailableError(RuntimeError):
    """Raised when local expert backend cannot be initialized or called safely."""


def _normalize_backend(value: Optional[str]) -> str:
    backend = str(value or "vllm").strip().lower()
    if backend in {"vllm", "local_hf", "arche_chem"}:
        return backend
    return backend


def _resolve_effective_model_config(model_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Priority: explicit args > ARCHE_CHEM_* env > existing defaults."""
    cfg = dict(DEFAULT_MODEL_CONFIG)

    env_model_path = os.getenv("ARCHE_CHEM_MODEL_PATH") or os.getenv("GAUSSIAN_LOCAL_MODEL_PATH")
    env_model_name = os.getenv("ARCHE_CHEM_MODEL_NAME") or os.getenv("GAUSSIAN_LOCAL_MODEL_NAME")
    env_backend = os.getenv("ARCHE_CHEM_BACKEND") or os.getenv("GAUSSIAN_LOCAL_BACKEND")

    if env_model_path:
        cfg["model"] = env_model_path
    if env_model_name:
        cfg["model_name"] = env_model_name
    if env_backend:
        cfg["backend"] = env_backend

    if model_config:
        cfg.update(model_config)

    if cfg.get("model_path") and not cfg.get("model"):
        cfg["model"] = cfg["model_path"]

    cfg["backend"] = _normalize_backend(cfg.get("backend"))
    return cfg


def _local_cache_key(model_config: Dict[str, Any]) -> str:
    return "|".join([
        str(model_config.get("backend", "vllm")),
        str(model_config.get("model", "")),
        str(model_config.get("parallel_size", 1)),
        str(model_config.get("gpu_memory_utilization", 0.7)),
    ])


def _load_vllm_backend() -> Dict[str, Any]:
    """Lazy-load vLLM runtime to keep module import safe on binary mismatch."""
    if _VLLM_RUNTIME:
        return _VLLM_RUNTIME
    try:
        vllm_module = importlib.import_module("vllm")
        llm_cls = getattr(vllm_module, "LLM")
        sampling_cls = getattr(vllm_module, "SamplingParams")
    except Exception as exc:
        raise LocalBackendUnavailableError(f"vLLM import failed: {exc}") from exc

    _VLLM_RUNTIME["LLM"] = llm_cls
    _VLLM_RUNTIME["SamplingParams"] = sampling_cls
    return _VLLM_RUNTIME


def _load_arche_chem_call_fn() -> Optional[Any]:
    """Best-effort load of shared ARCHE-Chem callable."""
    global _ARCHE_CHEM_CALL_FN, _ARCHE_CHEM_CALL_LOADED
    if _ARCHE_CHEM_CALL_LOADED:
        return _ARCHE_CHEM_CALL_FN

    _ARCHE_CHEM_CALL_LOADED = True
    candidates = [
        ("chemistry_multiagent.utils.arche_chem_client", "call_arche_chem"),
        ("utils.arche_chem_client", "call_arche_chem"),
        ("arche_chem_client", "call_arche_chem"),
    ]
    for module_name, attr in candidates:
        try:
            mod = importlib.import_module(module_name)
            fn = getattr(mod, attr, None)
            if callable(fn):
                _ARCHE_CHEM_CALL_FN = fn
                return _ARCHE_CHEM_CALL_FN
        except Exception:
            continue
    return None


def _try_shared_arche_chem_local_call(
    prompt: str,
    model_config: Dict[str, Any],
) -> str:
    call_fn = _load_arche_chem_call_fn()
    if call_fn is None:
        raise LocalBackendUnavailableError("shared ARCHE-Chem client is unavailable")
    try:
        text = call_fn(
            messages=[{"role": "user", "content": prompt}],
            model=model_config.get("model_name") or DEFAULT_API_MODEL,
            model_path=model_config.get("model"),
            backend=model_config.get("backend") or "local_hf",
            max_tokens=int(model_config.get("max_tokens", 1024)),
            temperature=float(model_config.get("temperature", 0.7)),
        )
    except Exception as exc:
        raise LocalBackendUnavailableError(f"shared ARCHE-Chem call failed: {exc}") from exc
    if not isinstance(text, str):
        text = str(text)
    return text


def _get_cached_local_llm(model_config: Dict[str, Any]) -> Any:
    backend = _normalize_backend(model_config.get("backend"))
    if backend not in {"vllm", "local_hf"}:
        raise LocalBackendUnavailableError(f"Unsupported local backend for vLLM path: {backend}")

    model_path = model_config.get("model")
    if not model_path:
        raise LocalBackendUnavailableError("Missing local model path")

    key = _local_cache_key(model_config)
    llm = _LOCAL_LLM_CACHE.get(key)
    if llm is None:
        runtime = _load_vllm_backend()
        llm_cls = runtime["LLM"]
        try:
            llm = llm_cls(
                model=model_path,
                gpu_memory_utilization=model_config.get("gpu_memory_utilization", 0.7),
                tensor_parallel_size=model_config.get("parallel_size", 1),
            )
        except Exception as exc:
            raise LocalBackendUnavailableError(f"vLLM local model init failed: {exc}") from exc
        _LOCAL_LLM_CACHE[key] = llm
    return llm


def _can_use_api(api_key: Optional[str]) -> bool:
    resolved_api_key = api_key or os.getenv("ARCHE_CHEM_API_KEY")
    return bool(resolved_api_key)


def _progress(step_log: List[str], message: str, verbose: bool = False) -> None:
    entry = f"[gen_gaussiancode] {message}"
    step_log.append(entry)
    if verbose:
        print(entry)


def _run_prompt_local_first(
    prompt: str,
    model_config: Dict[str, Any],
    api_key: Optional[str],
    api_base_url: str,
    api_model: str,
    timeout: int,
    stage_name: str,
    step_log: Optional[List[str]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Try local model first, then API fallback if available."""
    stage_result: Dict[str, Any] = {
        "text": "",
        "used_local_model": False,
        "used_api_fallback": False,
        "backend_used": None,
        "fallback_reason": None,
        "local_backend_error": None,
    }

    messages = [{"role": "user", "content": prompt}]
    logs = step_log if isinstance(step_log, list) else []
    _progress(logs, f"stage={stage_name} start", verbose=verbose)
    local_backend = _normalize_backend(model_config.get("backend"))
    if local_backend in {"local_hf", "arche_chem"}:
        try:
            _progress(logs, f"stage={stage_name} local_inference backend=arche_chem_client", verbose=verbose)
            local_text = _try_shared_arche_chem_local_call(prompt, model_config)
            stage_result["text"] = local_text or ""
            stage_result["used_local_model"] = True
            stage_result["backend_used"] = "arche_chem_client"
            if stage_result["text"].strip():
                _progress(logs, f"stage={stage_name} done via local", verbose=verbose)
                return stage_result
            stage_result["fallback_reason"] = f"{stage_name}: empty local output"
            _progress(logs, f"stage={stage_name} local_empty -> fallback", verbose=verbose)
        except Exception as exc:
            stage_result["local_backend_error"] = str(exc)
            stage_result["fallback_reason"] = f"{stage_name}: local call failed: {exc}"
            _progress(logs, f"stage={stage_name} local_failed: {exc}", verbose=verbose)

    if model_config.get("model") and local_backend in {"vllm", "local_hf", "arche_chem"}:
        try:
            _progress(logs, f"stage={stage_name} local_inference backend={local_backend}", verbose=verbose)
            local_answers = make_local_vllm_python_call(model_config, messages)
            stage_result["text"] = local_answers[0] if local_answers else ""
            stage_result["used_local_model"] = True
            stage_result["backend_used"] = "vllm"
            if stage_result["text"].strip():
                _progress(logs, f"stage={stage_name} done via local", verbose=verbose)
                return stage_result
            stage_result["fallback_reason"] = f"{stage_name}: empty local output"
            _progress(logs, f"stage={stage_name} local_empty -> fallback", verbose=verbose)
        except Exception as exc:
            stage_result["local_backend_error"] = str(exc)
            stage_result["fallback_reason"] = f"{stage_name}: local call failed: {exc}"
            _progress(logs, f"stage={stage_name} local_failed: {exc}", verbose=verbose)

    if _can_use_api(api_key):
        _progress(logs, f"stage={stage_name} api_fallback", verbose=verbose)
        api_text = make_api_call_with_retry(
            messages,
            api_key=api_key,
            base_url=api_base_url,
            model=api_model,
            timeout=timeout,
        )
        stage_result["text"] = api_text
        stage_result["used_api_fallback"] = True
        stage_result["backend_used"] = "api"
        _progress(logs, f"stage={stage_name} done via api", verbose=verbose)
        return stage_result

    if stage_result["fallback_reason"] is None:
        stage_result["fallback_reason"] = f"{stage_name}: no available backend"
    _progress(logs, f"stage={stage_name} failed: {stage_result['fallback_reason']}", verbose=verbose)
    return stage_result


def make_local_vllm_python_call(model_config: Dict[str, Any], messages: List[Dict[str, str]]) -> List[str]:
    """Use vLLM's Python API to call local model (cached instance)."""
    temperature = model_config.get("temperature", 0.7)
    max_tokens = model_config.get("max_tokens", 1024)
    input_text = "".join(msg["content"] for msg in messages)

    runtime = _load_vllm_backend()
    sampling_cls = runtime["SamplingParams"]
    sampling_params = sampling_cls(temperature=temperature, max_tokens=max_tokens)
    llm = _get_cached_local_llm(model_config)
    try:
        outputs = llm.generate(input_text, sampling_params)
    except Exception as exc:
        raise LocalBackendUnavailableError(f"vLLM local generate failed: {exc}") from exc
    return [output.text for output in outputs[0].outputs]


def make_api_call_with_retry(
    messages: List[Dict[str, str]],
    api_key: Optional[str] = None,
    base_url: str = DEFAULT_API_BASE_URL,
    model: str = DEFAULT_API_MODEL,
    timeout: int = DEFAULT_TIMEOUT,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    max_retries: int = 3,
    retry_delay: int = 5,
) -> str:
    """API call with retry/backoff."""
    resolved_api_key = api_key or os.getenv("ARCHE_CHEM_API_KEY")
    if not resolved_api_key:
        raise ValueError("Missing API key. Set ARCHE_CHEM_API_KEY or pass api_key.")

    openai.api_key = resolved_api_key
    openai.base_url = base_url

    local_retry_delay = retry_delay
    for attempt in range(max_retries):
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            return response.choices[0].message.content
        except Exception as exc:
            print(f"API call failed (attempt {attempt + 1}/{max_retries}): {exc}")
            if attempt < max_retries - 1:
                time.sleep(local_retry_delay)
                local_retry_delay *= 2
            else:
                raise


prompt_for_agent = """
You are a computational chemistry expert who specializes in preparing input files for Gaussian software.

Your task is to generate a complete Gaussian route section (the line that starts with `#`) based on the user's description of a quantum chemical calculation.

Guidelines:
- Use a single line starting with `#`, containing the method, basis set, and relevant keywords.
- Include geometry optimization (e.g., `opt`, `ts`, `calcfc`, etc.) if requested.
- Add frequency analysis (`freq`) if needed.
- Include solvent models (e.g., `scrf=(iefpcm,solvent=...)`) if specified.
- Set the temperature (e.g., `temperature=298.15`) if requested.
- Do not include any explanation — just output the complete route section.

Task:
{task_description}

Now generate the full Gaussian route section
"""

prompt_for_demand = """You are an expert computational chemist and clear communicator.
A user will provide a natural language description of their desired Gaussian calculation. The description may be vague or incomplete.

 **Description of the user's computational requirements:**  {User_input}

Your task is to:
    1. Analyze the user's input carefully and explain your understanding of their computational needs, including assumptions and detailed reasoning for each Gaussian keyword or parameter you choose to include.
    2. Based on your analysis, generate a detailed, professional Gaussian computational task description in English that could be directly used to prepare a Gaussian input file.

Your analysis should clearly justify the choice of:
    - The type of calculation (e.g., geometry optimization, transition state search, frequency analysis)
    - The functional and basis set (explain why you choose a specific method or a default)
    - Any special options or keywords (e.g., `calcfc`, `noeigentest`), explaining their purpose
    - Solvent model and solvent choice if applicable
    - Temperature or other relevant parameters if mentioned or inferred

Now, based on the following user input, provide the analysis and Gaussian computational task description:
Your output should strictly follow this format:
    Analysis: <Your detailed analysis of the user's request, including reasoning behind every selected keyword or parameter>
    Gaussian computational task description: <A clear, detailed computational task description suitable for Gaussian input generation>
"""

prompt_for_getcode = """
You are an expert in extracting Gaussian software inout code or keywords from text.
Given a text that may contain explanations, analysis, or other content, your task is to extract **only the Gaussian route section**, which:
    - Starts with a line beginning with the `#` character
    - Includes that line and any immediately following lines that are part of the Gaussian keywords/options
    - Does NOT include the title, charge, multiplicity, coordinates, or any other parts of the Gaussian input file

Return the extracted Gaussian route section as plain text, preserving line breaks exactly as in the original text.

If no route section is found, respond with:
"No Gaussian route section found."

Now, extract the Gaussian route section from the following text: {Text_to_extract_from}

Your output should strictly follow this format:
    Gaussian_code: <Gaussian_code>
"""

prompt_for_fixcode = """You are an expert computational chemist and clear communicator.
A user will provide:
  1. A natural language description of their desired Gaussian calculation.
  2. A raw Gaussian keyword line generated by another model, which may contain duplicates, redundancies, irrelevant, or invalid options.

 **Description of the user's computational requirements:** {User_input}
 **Raw Gaussian keyword line:** {Generated_keywords}

Your task is to:
    1. Analyze the user's input carefully and explain your understanding of their computational needs.
    2. Evaluate the raw Gaussian keyword line and explain your corrections.
    3. Produce a corrected Gaussian keyword line that is valid and concise.

Your output must strictly follow this format:
    Analysis: <Your detailed reasoning>
    Corrected Gaussian keyword line: <The final cleaned keyword line>
"""


def _parse_demand_response(demand_text: str) -> Dict[str, str]:
    if "Analysis:" in demand_text and "Gaussian computational task description:" in demand_text:
        parts = demand_text.split("Gaussian computational task description:", 1)
        return {
            "analysis": parts[0].replace("Analysis:", "").strip(),
            "task_description": parts[1].strip(),
        }
    return {
        "analysis": "Analysis failed",
        "task_description": "Generation failed",
    }


def _extract_gaussian_code(raw_text: str) -> str:
    if "Gaussian_code" in raw_text:
        return raw_text.split("Gaussian_code:", 1)[-1].strip()
    return raw_text.strip()


def _extract_fixed_gaussian_code(raw_text: str, fallback_code: str) -> str:
    if "Corrected Gaussian keyword line:" in raw_text:
        return raw_text.split("Corrected Gaussian keyword line:", 1)[-1].strip()
    return fallback_code


def generate_gaussian_code_result(
    user_input: str,
    model_config: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
    api_base_url: str = DEFAULT_API_BASE_URL,
    api_model: str = DEFAULT_API_MODEL,
    timeout: int = DEFAULT_TIMEOUT,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Name: generate_gaussian_code_result
    Description: Generate Gaussian route code with intermediate steps and structured output.
    Parameters:
    user_input: str Natural language Gaussian requirement.
    model_config: Optional[dict] Local vLLM config override.
    api_key: Optional[str] API key for remote model calls.
    api_base_url: str API base URL.
    api_model: str Remote model name.
    timeout: int API timeout seconds.
    Returns:
    dict Structured generation result.
    """
    result: Dict[str, Any] = {
        "success": False,
        "user_input": user_input,
        "demand_analysis": "",
        "task_description": "",
        "raw_model_answer": "",
        "raw_gaussian_code": "",
        "gaussian_code": "",
        "message": "",
        "metadata": {
            "backend_used": None,
            "model_name": None,
            "model_path": None,
            "used_local_model": False,
            "used_api_fallback": False,
            "fallback_reason": None,
            "local_backend_error": None,
            "stage_backends": {},
            "configured_backend": None,
            "processing_steps": [],
        },
    }

    step_log: List[str] = result["metadata"]["processing_steps"]
    if os.getenv("GAUSSIAN_CODE_VERBOSE", "").strip().lower() in {"1", "true", "yes", "on"}:
        verbose = True

    effective_model_config = _resolve_effective_model_config(model_config)
    _progress(step_log, f"config backend={effective_model_config.get('backend')} model_path={effective_model_config.get('model')}", verbose=verbose)
    result["metadata"]["configured_backend"] = effective_model_config.get("backend")
    result["metadata"]["model_name"] = effective_model_config.get("model_name")
    result["metadata"]["model_path"] = effective_model_config.get("model")

    try:
        demand_prompt = prompt_for_demand.format(User_input=user_input)
        demand_stage = _run_prompt_local_first(
            demand_prompt,
            effective_model_config,
            api_key=api_key,
            api_base_url=api_base_url,
            api_model=api_model,
            timeout=timeout,
            stage_name="demand",
            step_log=step_log,
            verbose=verbose,
        )
        demand_text = demand_stage.get("text", "")
        result["metadata"]["stage_backends"]["demand"] = demand_stage.get("backend_used")

        demand_parts = _parse_demand_response(demand_text)
        if demand_parts.get("task_description") in {"Generation failed", ""} and _can_use_api(api_key) and not demand_stage.get("used_api_fallback"):
            _progress(step_log, "stage=demand force_api_retry_for_parse", verbose=verbose)
            demand_text = make_api_call_with_retry(
                [{"role": "user", "content": demand_prompt}],
                api_key=api_key,
                base_url=api_base_url,
                model=api_model,
                timeout=timeout,
            )
            demand_parts = _parse_demand_response(demand_text)
            result["metadata"]["stage_backends"]["demand"] = "api"
            demand_stage["used_api_fallback"] = True

        result["demand_analysis"] = demand_parts["analysis"]
        result["task_description"] = demand_parts["task_description"] if demand_parts["task_description"] != "Generation failed" else user_input

        agent_prompt = prompt_for_agent.format(task_description=result["task_description"])
        agent_stage = _run_prompt_local_first(
            agent_prompt,
            effective_model_config,
            api_key=api_key,
            api_base_url=api_base_url,
            api_model=api_model,
            timeout=timeout,
            stage_name="agent",
            step_log=step_log,
            verbose=verbose,
        )
        result["raw_model_answer"] = (agent_stage.get("text") or "").strip()
        result["metadata"]["stage_backends"]["agent"] = agent_stage.get("backend_used")

        extract_prompt = prompt_for_getcode.format(Text_to_extract_from=result["raw_model_answer"])
        extract_stage = _run_prompt_local_first(
            extract_prompt,
            effective_model_config,
            api_key=api_key,
            api_base_url=api_base_url,
            api_model=api_model,
            timeout=timeout,
            stage_name="extract",
            step_log=step_log,
            verbose=verbose,
        )
        result["metadata"]["stage_backends"]["extract"] = extract_stage.get("backend_used")
        result["raw_gaussian_code"] = _extract_gaussian_code(extract_stage.get("text", ""))

        if result["raw_gaussian_code"] and "#" not in result["raw_gaussian_code"] and _can_use_api(api_key) and not extract_stage.get("used_api_fallback"):
            _progress(step_log, "stage=extract force_api_retry_for_route", verbose=verbose)
            raw_code_text = make_api_call_with_retry(
                [{"role": "user", "content": extract_prompt}],
                api_key=api_key,
                base_url=api_base_url,
                model=api_model,
                timeout=timeout,
            )
            result["raw_gaussian_code"] = _extract_gaussian_code(raw_code_text)
            result["metadata"]["stage_backends"]["extract"] = "api"
            extract_stage["used_api_fallback"] = True

        fix_prompt = prompt_for_fixcode.format(
            User_input=result["task_description"],
            Generated_keywords=result["raw_gaussian_code"],
        )
        fix_stage = _run_prompt_local_first(
            fix_prompt,
            effective_model_config,
            api_key=api_key,
            api_base_url=api_base_url,
            api_model=api_model,
            timeout=timeout,
            stage_name="fix",
            step_log=step_log,
            verbose=verbose,
        )
        result["metadata"]["stage_backends"]["fix"] = fix_stage.get("backend_used")
        result["gaussian_code"] = _extract_fixed_gaussian_code(
            fix_stage.get("text", ""),
            result["raw_gaussian_code"],
        )

        if not result["gaussian_code"] and result["raw_gaussian_code"]:
            result["gaussian_code"] = result["raw_gaussian_code"]

        used_local = any(bool(stage.get("used_local_model")) for stage in [demand_stage, agent_stage, extract_stage, fix_stage])
        used_api_fallback = any(bool(stage.get("used_api_fallback")) for stage in [demand_stage, agent_stage, extract_stage, fix_stage])
        fallback_reasons = [stage.get("fallback_reason") for stage in [demand_stage, agent_stage, extract_stage, fix_stage] if stage.get("fallback_reason")]
        local_backend_errors = [stage.get("local_backend_error") for stage in [demand_stage, agent_stage, extract_stage, fix_stage] if stage.get("local_backend_error")]

        stage_backends = result["metadata"].get("stage_backends", {})
        backend_used = "local" if any(v in {"vllm", "local_hf", "arche_chem_client"} for v in stage_backends.values()) else "api"
        if used_local and used_api_fallback:
            backend_used = "local_with_api_fallback"

        result["metadata"].update(
            {
                "backend_used": backend_used,
                "used_local_model": used_local,
                "used_api_fallback": used_api_fallback,
                "fallback_reason": "; ".join(fallback_reasons) if fallback_reasons else None,
                "local_backend_error": "; ".join(local_backend_errors) if local_backend_errors else None,
            }
        )

        if not result["gaussian_code"]:
            result["message"] = "Failed to generate Gaussian code"
            _progress(step_log, "final failed: empty gaussian_code", verbose=verbose)
            return result

        result["success"] = True
        result["message"] = "Gaussian code generated successfully"
        _progress(step_log, "final success", verbose=verbose)
        return result
    except Exception as exc:
        result["message"] = str(exc)
        if not result.get("metadata"):
            result["metadata"] = {}
        result["metadata"].setdefault("backend_used", "unknown")
        _progress(result.get("metadata", {}).setdefault("processing_steps", []), f"final exception: {exc}", verbose=verbose)
        return result

def generate_gaussian_code(
    user_input: str,
    model_config: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
    api_base_url: str = DEFAULT_API_BASE_URL,
    api_model: str = DEFAULT_API_MODEL,
    timeout: int = DEFAULT_TIMEOUT,
    verbose: bool = False,
) -> str:
    """Compatibility wrapper returning only final Gaussian route code."""
    result = generate_gaussian_code_result(
        user_input=user_input,
        model_config=model_config,
        api_key=api_key,
        api_base_url=api_base_url,
        api_model=api_model,
        timeout=timeout,
        verbose=verbose,
    )
    if not result["success"]:
        raise RuntimeError(result["message"])
    return result["gaussian_code"]


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Gaussian route code from natural language")
    parser.add_argument("--user_input", required=True, help="Natural language Gaussian requirement")
    parser.add_argument("--api_key", default=None, help="API key (optional if env is set)")
    parser.add_argument("--api_base_url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--api_model", default=DEFAULT_API_MODEL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--local_model_path", default=None, help="Override local vLLM model path")
    parser.add_argument("--local_backend", default=None, help="Override local backend (vllm/local_hf)")
    parser.add_argument("--local_model_name", default=None, help="Override local model name metadata")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", help="Print step-by-step progress")
    return parser


def main() -> None:
    args = _build_cli_parser().parse_args()

    override_model_config: Dict[str, Any] = {}
    if args.local_model_path:
        override_model_config["model"] = args.local_model_path
    if args.local_backend:
        override_model_config["backend"] = args.local_backend
    if args.local_model_name:
        override_model_config["model_name"] = args.local_model_name
    if args.temperature is not None:
        override_model_config["temperature"] = args.temperature
    if args.max_tokens is not None:
        override_model_config["max_tokens"] = args.max_tokens

    output = generate_gaussian_code_result(
        user_input=args.user_input,
        model_config=override_model_config or None,
        api_key=args.api_key,
        api_base_url=args.api_base_url,
        api_model=args.api_model,
        timeout=args.timeout,
        verbose=args.verbose,
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
