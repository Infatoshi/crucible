"""LLM API communication layer. Supports Anthropic native + OpenAI/OpenRouter."""

import json
import time
from typing import Tuple

from src.models import ModelConfig
from src.tools import TOOLS_OPENAI

# Anthropic-format tool schemas
TOOLS_ANTHROPIC = [
    {
        "name": t["function"]["name"],
        "description": t["function"]["description"],
        "input_schema": t["function"]["parameters"],
    }
    for t in TOOLS_OPENAI
]

MAX_RETRIES = 8
RETRY_DELAY = 15


def get_model_response(client, model_config: ModelConfig, system_prompt: str, messages: list):
    """Call the LLM with retries. Returns raw response."""
    for attempt in range(MAX_RETRIES):
        try:
            if model_config.provider == "anthropic":
                resp = _call_anthropic(client, model_config, system_prompt, messages)
            else:
                resp = _call_openai(client, model_config, system_prompt, messages)

            # Check for OpenRouter-style error in response body (choices=None)
            if model_config.provider != "anthropic" and not getattr(resp, "choices", None):
                error = getattr(resp, "error", None)
                error_msg = str(error) if error else "empty choices"
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_DELAY * (attempt + 1)
                    print(f"    API error (attempt {attempt+1}): {error_msg}. Retrying in {wait}s...", flush=True)
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"API returned no choices after {MAX_RETRIES} attempts: {error_msg}")

            return resp

        except Exception as e:
            err = str(e).lower()
            retryable = "overloaded" in err or "502" in err or "529" in err or "rate" in err or "timeout" in err
            if attempt < MAX_RETRIES - 1 and retryable:
                wait = RETRY_DELAY * (attempt + 1)
                print(f"    API error (attempt {attempt+1}): {e}. Retrying in {wait}s...", flush=True)
                time.sleep(wait)
            else:
                raise


def _call_anthropic(client, model_config: ModelConfig, system_prompt: str, messages: list):
    """Call Anthropic Messages API."""
    return client.messages.create(
        model=model_config.model_id,
        max_tokens=16384,
        system=system_prompt,
        messages=messages,
        tools=TOOLS_ANTHROPIC,
    )


def _call_openai(client, model_config: ModelConfig, system_prompt: str, messages: list):
    """Call OpenAI-compatible API (OpenAI direct or OpenRouter)."""
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    kwargs = {
        "model": model_config.model_id,
        "max_tokens": 16384,
        "messages": full_messages,
        "tools": TOOLS_OPENAI,
    }
    return client.chat.completions.create(**kwargs)


def parse_response(response, model_config: ModelConfig) -> Tuple[str, list]:
    """Extract text content and tool calls from response."""
    if model_config.provider == "anthropic":
        return _parse_anthropic(response)
    else:
        return _parse_openai(response)


def _parse_anthropic(response) -> Tuple[str, list]:
    """Parse Anthropic Messages API response."""
    content = ""
    tool_calls = []
    for block in response.content:
        if block.type == "text":
            content += block.text
        elif block.type == "tool_use":
            tool_calls.append({
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })
    return content, tool_calls


def _parse_openai(response) -> Tuple[str, list]:
    """Parse OpenAI-compatible response."""
    if not response.choices:
        error_msg = getattr(response, "error", "unknown")
        raise RuntimeError(f"API returned no choices: {error_msg}")
    message = response.choices[0].message
    content = message.content or ""
    tool_calls = []
    if message.tool_calls:
        for tc in message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                # Try to salvage malformed JSON
                args = {"command": tc.function.arguments} if tc.function.name == "bash" else {}
            tool_calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "input": args,
            })
    return content, tool_calls


def extract_token_usage(response, model_config: ModelConfig) -> Tuple[int, int]:
    """Return (input_tokens, output_tokens)."""
    if model_config.provider == "anthropic":
        if hasattr(response, "usage") and response.usage:
            return (
                getattr(response.usage, "input_tokens", 0),
                getattr(response.usage, "output_tokens", 0),
            )
    else:
        if hasattr(response, "usage") and response.usage:
            return (
                getattr(response.usage, "prompt_tokens", 0),
                getattr(response.usage, "completion_tokens", 0),
            )
    return 0, 0


def format_assistant_message(content: str, tool_calls: list, model_config: ModelConfig) -> dict:
    """Format assistant message for conversation history."""
    if model_config.provider == "anthropic":
        blocks = []
        if content:
            blocks.append({"type": "text", "text": content})
        for tc in tool_calls:
            blocks.append({
                "type": "tool_use",
                "id": tc["id"],
                "name": tc["name"],
                "input": tc["input"],
            })
        return {"role": "assistant", "content": blocks}
    else:
        msg: dict = {"role": "assistant"}
        if content:
            msg["content"] = content
        if tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["input"]),
                    },
                }
                for tc in tool_calls
            ]
        return msg


def format_tool_results(tool_results: list, model_config: ModelConfig) -> list:
    """Format tool results for conversation history."""
    if model_config.provider == "anthropic":
        return [{
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tr["id"],
                    "content": tr["content"],
                }
                for tr in tool_results
            ],
        }]
    else:
        return [
            {"role": "tool", "tool_call_id": tr["id"], "content": tr["content"]}
            for tr in tool_results
        ]
