"""Tool schemas and dispatch for sandboxed agents."""

import re
from typing import Any, Dict, List

BLOCKED_COMMANDS = re.compile(
    r"(?:^|\s|;|&&|\|\|)"
    r"(?:pkill|killall|kill\s+-9"
    r"|rm\s+-rf\s+/(?:$|\s|;)|rm\s+-rf\s+/(?:workspace|home|usr|etc|var)\b"
    r"|chmod|chown"
    r"|nvidia-smi\s+--(?:reset|drain|persist)"
    r")",
    re.IGNORECASE,
)

BLOCKED_WRITE_PATHS = {"_benchmark.py", "reference.py", "_evaluate.py"}


def dispatch_tool(tool_name: str, tool_input: dict, sandbox) -> str:
    if tool_name == "read_file":
        path = tool_input.get("path", "")
        content = sandbox.read_file(path)
        if content is None:
            return f"Error: file not found: {path}"
        offset = tool_input.get("offset")
        limit = tool_input.get("limit")
        if offset or limit:
            lines = content.splitlines(keepends=True)
            start = max((offset or 1) - 1, 0)
            end = start + (limit or len(lines))
            return "".join(lines[start:end])
        return content

    if tool_name == "write_file":
        path = tool_input.get("path", "")
        basename = path.rsplit("/", 1)[-1] if "/" in path else path
        if basename in BLOCKED_WRITE_PATHS:
            return f"Error: cannot overwrite protected file: {path}"
        content = tool_input.get("content", "")
        success = sandbox.write_file(path, content)
        return f"Successfully wrote to {path}" if success else f"Error writing file: {path}"

    if tool_name == "edit_file":
        path = tool_input.get("path", "")
        basename = path.rsplit("/", 1)[-1] if "/" in path else path
        if basename in BLOCKED_WRITE_PATHS:
            return f"Error: cannot edit protected file: {path}"
        old_str = tool_input.get("old_str", "")
        new_str = tool_input.get("new_str", "")
        content = sandbox.read_file(path)
        if content is None:
            return f"Error: file not found: {path}"
        if old_str not in content:
            preview = content[:500] + "..." if len(content) > 500 else content
            return f"Error: old_str not found in {path}. File content:\n{preview}"
        count = content.count(old_str)
        if count > 1:
            return f"Error: old_str matches {count} locations in {path} -- must be unique"
        new_content = content.replace(old_str, new_str, 1)
        sandbox.write_file(path, new_content)
        return f"Applied edit to {path}"

    if tool_name == "bash":
        cmd = tool_input.get("command", "")
        if BLOCKED_COMMANDS.search(cmd):
            return "Error: command blocked by security policy."
        timeout = tool_input.get("timeout")
        result = sandbox.run_command(cmd, timeout=timeout) if timeout else sandbox.run_command(cmd)
        return (
            f"stdout:\n{result['stdout']}\n"
            f"stderr:\n{result['stderr']}\n"
            f"return_code: {result['returncode']}"
        )

    if tool_name == "submit":
        return f"Submitted: {tool_input.get('solution_path', 'solution.py')}"

    return f"Error: unknown tool '{tool_name}'"


# OpenAI-compatible tool schemas (used by OpenRouter for all models)
TOOLS_OPENAI: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to workspace"},
                    "offset": {"type": "integer", "description": "Start line (1-indexed)"},
                    "limit": {"type": "integer", "description": "Number of lines to read"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "File content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace a unique string in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "old_str": {"type": "string", "description": "String to find (must be unique)"},
                    "new_str": {"type": "string", "description": "Replacement string"},
                },
                "required": ["path", "old_str", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a shell command in the sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submit your solution for evaluation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "solution_path": {
                        "type": "string",
                        "description": "Path to solution file (default: solution.py)",
                    },
                },
                "required": [],
            },
        },
    },
]
