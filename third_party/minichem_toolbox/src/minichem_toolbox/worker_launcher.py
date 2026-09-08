"""Apply worker resource limits before loading scientific backend modules."""

from __future__ import annotations

import os
import resource
import runpy
import signal


def _interrupt_worker(_signum, _frame) -> None:
    """Turn evaluator cancellation into unwindable Python control flow."""

    raise KeyboardInterrupt


def _apply_resource_limits() -> None:
    if os.environ.get("MINICHEM_BACKEND_RUNTIME") == "multiwfn":
        _stack_soft, stack_hard = resource.getrlimit(resource.RLIMIT_STACK)
        resource.setrlimit(resource.RLIMIT_STACK, (stack_hard, stack_hard))

    cpu_ids = [
        int(item)
        for item in os.environ.get("MINICHEM_WORKER_CPU_IDS", "").split(",")
        if item.strip()
    ]
    if cpu_ids and hasattr(os, "sched_setaffinity"):
        allowed = set(os.sched_getaffinity(0))
        selected = set(cpu_ids)
        if not selected <= allowed:
            raise RuntimeError(
                "Allocated CPU ids are outside the worker affinity: "
                f"allocated={sorted(selected)}, allowed={sorted(allowed)}"
            )
        os.sched_setaffinity(0, selected)
    elif hasattr(os, "sched_getaffinity"):
        cpu_count = int(os.environ.get("MINICHEM_WORKER_CPU_COUNT", "0") or 0)
        if cpu_count > 0:
            allowed = sorted(os.sched_getaffinity(0))
            os.sched_setaffinity(0, set(allowed[:cpu_count]))

def main() -> int:
    _apply_resource_limits()
    signal.signal(signal.SIGTERM, _interrupt_worker)
    signal.signal(signal.SIGINT, _interrupt_worker)
    runpy.run_module("minichem_toolbox.worker", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
