#!/usr/bin/env python3
"""Overnight tournament runner. Runs all 5 protocols sequentially, Claude vs Codex."""

import os
import time
import traceback
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

from src.orchestrator import Tournament  # noqa: E402
from src.protocols import get_protocol  # noqa: E402

AGENTS = ["claude", "codex"]
ROUNDS = 10
TURNS = 15
TIMEOUT = 1200  # 20 min per agent per round

PROTOCOLS = [
    ("kernel", {"levels": [1, 2]}),
    ("quantization", {}),
    ("interp", {}),
    ("scaling", {}),
    ("reward_hacking", {}),
]


def run_all():
    results = []
    for protocol_name, kwargs in PROTOCOLS:
        print(f"\n\n{'#'*70}", flush=True)
        print(f"# STARTING PROTOCOL: {protocol_name}", flush=True)
        print(f"{'#'*70}\n", flush=True)

        try:
            protocol = get_protocol(protocol_name, **kwargs)
            tournament = Tournament(
                protocol=protocol,
                agent_keys=AGENTS,
                max_rounds=ROUNDS,
                max_turns_per_agent=TURNS,
                timeout_per_agent=TIMEOUT,
            )
            summary = tournament.run()
            results.append({"protocol": protocol_name, "status": "completed", "summary": summary})
        except Exception as e:
            print(f"\nPROTOCOL {protocol_name} FAILED: {e}", flush=True)
            traceback.print_exc()
            results.append({"protocol": protocol_name, "status": "failed", "error": str(e)})

        # Brief pause between protocols
        time.sleep(5)

    # Print final summary
    print(f"\n\n{'='*70}", flush=True)
    print("OVERNIGHT RUN COMPLETE", flush=True)
    print(f"{'='*70}", flush=True)
    for r in results:
        if r["status"] == "completed":
            s = r["summary"]
            print(f"  {r['protocol']:20s}  Champion: {s['best_agent']}  Fitness: {s['best_fitness']}")
        else:
            print(f"  {r['protocol']:20s}  FAILED: {r['error'][:60]}")
    print(flush=True)


if __name__ == "__main__":
    run_all()
