#!/usr/bin/env python3
"""Run quantization tournament: 9 agents on Qwen3-4B.

Claude Code, Codex CLI, and 7 models via Cursor Agent CLI.
Hardened protocol: hidden eval texts, perplexity sanity checks.
"""

import os
from pathlib import Path

# Load env vars
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
                os.environ.setdefault(key, value)

from src.orchestrator import Tournament
from src.protocols import get_protocol

AGENTS = [
    "claude",        # Claude Code CLI (Opus 4.6)
    "codex",         # Codex CLI (GPT 5.4)
    "composer2",     # Cursor Composer 2
    "gpt54xh",       # GPT-5.4 XHigh via Cursor
    "gpt53codex",    # GPT-5.3 Codex XHigh via Cursor
    "gemini31",      # Gemini 3.1 Pro via Cursor
    "grok4",         # Grok 4.20 via Cursor
    "opus46cursor",  # Opus 4.6 via Cursor (compare harness effect)
    "sonnet46",      # Sonnet 4.6 via Cursor
]

protocol = get_protocol("quantization")
tournament = Tournament(
    protocol=protocol,
    agent_keys=AGENTS,
    max_rounds=5,          # fewer rounds, more agents
    max_turns_per_agent=15,
    timeout_per_agent=2400,
)
summary = tournament.run()
