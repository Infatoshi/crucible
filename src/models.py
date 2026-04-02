"""Model configurations for CLI-based agents."""

from dataclasses import dataclass
from typing import Dict, Literal


@dataclass
class ModelConfig:
    name: str
    model_id: str
    provider: Literal["cli-claude", "cli-codex"]


MODELS: Dict[str, ModelConfig] = {
    # Native CLIs
    "claude": ModelConfig(
        name="Claude Code (Opus 4.6)",
        model_id="claude-opus-4-6",
        provider="cli-claude",
    ),
    "codex": ModelConfig(
        name="Codex CLI (GPT 5.4)",
        model_id="codex",
        provider="cli-codex",
    ),
    # Cursor Agent models
    "composer2": ModelConfig(
        name="Cursor Composer 2",
        model_id="composer-2",
        provider="cli-cursor",
    ),
    "gpt54xh": ModelConfig(
        name="GPT-5.4 XHigh",
        model_id="gpt-5.4-xhigh",
        provider="cli-cursor",
    ),
    "gpt53codex": ModelConfig(
        name="GPT-5.3 Codex XHigh",
        model_id="gpt-5.3-codex-xhigh",
        provider="cli-cursor",
    ),
    "gemini31": ModelConfig(
        name="Gemini 3.1 Pro",
        model_id="gemini-3.1-pro",
        provider="cli-cursor",
    ),
    "grok4": ModelConfig(
        name="Grok 4.20",
        model_id="grok-4-20",
        provider="cli-cursor",
    ),
    "opus46cursor": ModelConfig(
        name="Opus 4.6 (via Cursor)",
        model_id="claude-4.6-opus-high",
        provider="cli-cursor",
    ),
    "sonnet46": ModelConfig(
        name="Sonnet 4.6",
        model_id="claude-4.6-sonnet-medium",
        provider="cli-cursor",
    ),
}


def get_model_config(key: str) -> ModelConfig:
    if key not in MODELS:
        raise ValueError(f"Unknown model: {key}. Available: {list(MODELS.keys())}")
    return MODELS[key]
