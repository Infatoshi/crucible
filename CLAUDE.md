# Crucible

Multi-agent adversarial research swarm. Tournament topology where N agents compete,
GPU benchmarks are the judge, and the best solution seeds the next round.

## Quick Commands
```bash
uv sync
uv pip install bitsandbytes autoawq transformers accelerate datasets

# Run quantization tournament
uv run python run_quant.py

# Run all 5 protocols overnight
uv run python run_overnight.py
```

## Key files
- `src/agent.py` -- CLI runners for Claude Code, Codex, Cursor Agent
- `src/models.py` -- Model registry (add new models here)
- `src/orchestrator.py` -- Tournament engine
- `src/protocols/quantization.py` -- The Qwen3-4B quantization protocol
- `run_quant.py` -- Main tournament runner (edit agents/rounds here)

## API keys needed
Set in environment or `~/.env_vars`:
- `ANTHROPIC_API_KEY` -- Claude Code
- `OPENAI_API_KEY` -- Codex CLI
- `CURSOR_API_KEY` -- Cursor Agent
