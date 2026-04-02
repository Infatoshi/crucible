"""Agent execution via CLI tools (Claude Code, Codex CLI)."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

from src.models import ModelConfig
from src.run_log import AgentResult, RunLog
from src.sandbox import Sandbox

CODEX_BIN = os.path.expanduser("~/.local/node-v22.14.0-linux-x64/bin/codex")
CURSOR_BIN = os.path.expanduser("~/.local/bin/agent")


def run_agent(
    model_config: ModelConfig,
    sandbox: Sandbox,
    system_prompt: str,
    initial_message: str,
    max_turns: int = 15,
    timeout_seconds: int = 600,
    run_log: Optional[RunLog] = None,
    agent_name: str = "",
    round_num: int = 0,
    protocol_name: str = "",
) -> AgentResult:
    """Run an agent CLI in the sandbox workspace. Returns AgentResult."""
    t0 = time.time()
    workspace = str(sandbox.workspace_path)

    # Combine system prompt and user message
    prompt = f"{system_prompt}\n\n---\n\n{initial_message}"

    if model_config.provider == "cli-claude":
        result = _run_claude_cli(workspace, prompt, timeout_seconds, agent_name)
    elif model_config.provider == "cli-codex":
        result = _run_codex_cli(workspace, prompt, timeout_seconds, agent_name)
    elif model_config.provider == "cli-cursor":
        result = _run_cursor_cli(workspace, prompt, timeout_seconds, agent_name, model_config.model_id)
    else:
        raise ValueError(f"Unknown CLI provider: {model_config.provider}")

    elapsed = time.time() - t0

    # Save CLI output as artifact
    if run_log:
        run_log.write_turn_artifact(agent_name, round_num, 1, "cli_output.txt", result["stdout"])
        if result["stderr"]:
            run_log.write_turn_artifact(agent_name, round_num, 1, "cli_stderr.txt", result["stderr"])

    # Check if solution.py was created
    submitted = sandbox.file_exists("solution.py")
    solution_code = None
    if submitted:
        solution_code = sandbox.read_file("solution.py")
        print(f"  [{agent_name}] Solution written ({len(solution_code)} chars)", flush=True)
    else:
        print(f"  [{agent_name}] No solution.py produced", flush=True)

    # Parse metadata from JSON output (Claude CLI)
    num_turns = 0
    if result.get("json_result"):
        jr = result["json_result"]
        num_turns = jr.get("num_turns", 0)
        _ = jr.get("total_cost_usd", 0.0)  # logged in JSON output

    error = None
    if result["returncode"] != 0 and not submitted:
        error = f"CLI exited with code {result['returncode']}: {result['stderr'][-500:]}"

    return AgentResult(
        agent=agent_name,
        model_id=model_config.model_id,
        round_num=round_num,
        protocol=protocol_name,
        turns_used=num_turns,
        submitted=submitted,
        solution_code=solution_code,
        error=error,
        elapsed_seconds=elapsed,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def _run_claude_cli(workspace: str, prompt: str, timeout: int, agent_name: str) -> dict:
    """Run Claude Code CLI in print mode with JSON output."""
    print(f"  [{agent_name}] Launching Claude Code CLI...", flush=True)

    cmd = [
        "claude",
        "--print",
        "--dangerously-skip-permissions",
        "--output-format", "json",
        "--model", "opus",
        "--max-budget-usd", "50",
        prompt,
    ]

    result = _exec_cli(cmd, workspace, timeout, agent_name)

    # Parse JSON result
    stdout = result["stdout"].strip()
    if stdout:
        try:
            result["json_result"] = json.loads(stdout)
            jr = result["json_result"]
            print(
                f"  [{agent_name}] Claude: {jr.get('num_turns', '?')} turns, "
                f"${jr.get('total_cost_usd', 0):.3f}, "
                f"stop={jr.get('stop_reason', '?')}",
                flush=True,
            )
        except json.JSONDecodeError:
            result["json_result"] = None

    return result


def _run_codex_cli(workspace: str, prompt: str, timeout: int, agent_name: str) -> dict:
    """Run Codex CLI in exec mode."""
    print(f"  [{agent_name}] Launching Codex CLI...", flush=True)

    cmd = [
        CODEX_BIN,
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "--cd", workspace,
        "--skip-git-repo-check",
        "--ephemeral",
        prompt,
    ]

    return _exec_cli(cmd, workspace, timeout, agent_name)


def _run_cursor_cli(workspace: str, prompt: str, timeout: int, agent_name: str, model_id: str) -> dict:
    """Run Cursor Agent CLI in print mode."""
    print(f"  [{agent_name}] Launching Cursor Agent CLI (model={model_id})...", flush=True)

    # Use stream-json: some models return empty with plain json format,
    # but stream-json always emits a final result line we can parse.
    cmd = [
        CURSOR_BIN,
        "--print",
        "--force",
        "--output-format", "stream-json",
        "--model", model_id,
        "--workspace", workspace,
        prompt,
    ]

    result = _exec_cli(cmd, workspace, timeout, agent_name)

    # Parse the last JSON line with type=result from stream-json output
    stdout = result["stdout"].strip()
    if stdout:
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                if parsed.get("type") == "result":
                    result["json_result"] = parsed
                    usage = parsed.get("usage", {})
                    print(
                        f"  [{agent_name}] Cursor({model_id}): "
                        f"{usage.get('outputTokens', '?')} out tokens, "
                        f"stop={parsed.get('subtype', '?')}, "
                        f"{parsed.get('duration_ms', 0) / 1000:.1f}s",
                        flush=True,
                    )
                    break
            except json.JSONDecodeError:
                continue

    return result


def _load_env() -> dict:
    """Load environment with API keys from ~/.env_vars."""
    env = os.environ.copy()
    env_file = Path.home() / ".env_vars"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                if line.startswith("export "):
                    line = line[7:]
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and value:
                    env.setdefault(key, value)
    return env


def _exec_cli(cmd: list, workspace: str, timeout: int, agent_name: str) -> dict:
    """Execute a CLI command, capture output."""
    env = _load_env()

    try:
        completed = subprocess.run(
            cmd,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        return {
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "returncode": completed.returncode,
            "json_result": None,
        }
    except subprocess.TimeoutExpired as e:
        print(f"  [{agent_name}] CLI timed out after {timeout}s", flush=True)
        return {
            "stdout": e.stdout or "",
            "stderr": e.stderr or "",
            "returncode": -1,
            "json_result": None,
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
            "json_result": None,
        }
