#!/usr/bin/env python3
"""Create one isolated Agent workspace and its local MCP configurations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toolbox-root", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--model", default="deepseek/deepseek-v4-flash")
    parser.add_argument("--base-url", default="https://api.deepseek.com/v1")
    args = parser.parse_args()

    root = Path(args.toolbox_root).resolve()
    workspace = Path(args.workspace).resolve()
    for name in (
        "code",
        "data",
        "outputs",
        "report",
        "tool_logs",
        "_tool_results",
        "_tool_artifacts",
        "_sessions",
    ):
        (workspace / name).mkdir(parents=True, exist_ok=True)

    prompt = root / "tests" / "harnesses" / "PROMPT.md"
    (workspace / "INSTRUCTIONS.md").write_text(prompt.read_text(encoding="utf-8"), encoding="utf-8")

    launcher = workspace / "start_minichem_mcp.sh"
    launcher.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"export MINICHEM_HOME={json.dumps(str(root))}\n"
        f"export MINICHEM_MCP_WORKSPACE={json.dumps(str(workspace))}\n"
        f"export RESEARCHCHEMBENCH_WORKSPACE={json.dumps(str(workspace))}\n"
        f"export RESEARCHCHEMBENCH_RUN_ID={json.dumps(workspace.name)}\n"
        f"export CHEMGRAPH_LOG_DIR={json.dumps(str(workspace / 'tool_logs'))}\n"
        "export RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES=4\n"
        "export RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB=8192\n"
        "export RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT=0\n"
        f"exec {json.dumps(str(root / 'scripts' / 'start_mcp.sh'))} "
        "--transport stdio --discovery-mode progressive\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)

    mcp_server = {
        "type": "stdio",
        "command": str(launcher),
        "args": [],
        "env": {},
    }
    write_json(workspace / ".mcp.json", {"mcpServers": {"minichem_toolbox": mcp_server}})

    provider, model_id = args.model.split("/", 1) if "/" in args.model else ("deepseek", args.model)
    qualified_model = f"{provider}/{model_id}"
    write_json(
        workspace / "opencode.json",
        {
            "$schema": "https://opencode.ai/config.json",
            "model": qualified_model,
            "small_model": qualified_model,
            "provider": {
                provider: {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": provider,
                    "options": {
                        "baseURL": args.base_url,
                        "apiKey": "{env:OPENAI_API_KEY}",
                    },
                    "models": {model_id: {"name": model_id}},
                }
            },
            "agent": {"build": {"steps": 80}, "general": {"steps": 80}},
            "mcp": {
                "minichem_toolbox": {
                    "type": "local",
                    "command": [str(launcher)],
                    "environment": {},
                    "timeout": 3_600_000,
                    "enabled": True,
                }
            },
        },
    )
    write_json(
        workspace / "harness_meta.json",
        {
            "toolbox_root": os.path.relpath(root, workspace),
            "model": qualified_model,
            "base_url": args.base_url,
            "mcp_server": "minichem_toolbox",
            "prompt": "INSTRUCTIONS.md",
        },
    )
    print(workspace)


if __name__ == "__main__":
    main()
