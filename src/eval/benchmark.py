"""GPU benchmark utilities shared across protocols."""

from __future__ import annotations

import json
from typing import Any, Dict

from src.sandbox import Sandbox


def run_benchmark_script(sandbox: Sandbox, script_name: str = "_benchmark.py", timeout: int = 120) -> Dict[str, Any]:
    """Run a benchmark script in the sandbox and parse JSON output."""
    result = sandbox.run_command(f"python {script_name}", timeout=timeout)
    stdout = result["stdout"].strip()

    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    return {
        "error": f"Could not parse output: {stdout[:500]}",
        "stderr": result.get("stderr", "")[:500],
        "returncode": result.get("returncode"),
    }
