#!/usr/bin/env python3
"""Crucible: Multi-agent adversarial research swarm."""

import argparse
import json
import os
from pathlib import Path


def load_env_vars():
    """Load API keys from ~/.env_vars if it exists."""
    env_file = Path.home() / ".env_vars"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                # Strip 'export ' prefix if present
                if line.startswith("export "):
                    line = line[7:]
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and value:
                    os.environ.setdefault(key, value)


def cmd_run(args):
    """Run a tournament."""
    from src.orchestrator import Tournament
    from src.protocols import get_protocol

    protocol_kwargs = {}
    if args.problem:
        protocol_kwargs["problem_path"] = args.problem
    if args.levels:
        protocol_kwargs["levels"] = [int(x) for x in args.levels.split(",")]

    protocol = get_protocol(args.protocol, **protocol_kwargs)
    agents = [a.strip() for a in args.agents.split(",")]

    output_dir = Path(args.output) if args.output else None

    tournament = Tournament(
        protocol=protocol,
        agent_keys=agents,
        max_rounds=args.rounds,
        max_turns_per_agent=args.turns,
        timeout_per_agent=args.timeout,
        output_dir=output_dir,
    )

    summary = tournament.run()
    return summary


def cmd_list(args):
    """List available protocols and agents."""
    from src.models import MODELS
    from src.protocols import PROTOCOLS

    print("\nProtocols:")
    for name, cls in PROTOCOLS.items():
        p = cls()
        print(f"  {name:20s}  fitness={p.fitness_key} ({p.fitness_direction})")

    print("\nAgents:")
    for key, config in MODELS.items():
        print(f"  {key:20s}  {config.name} ({config.provider})")

    print()


def cmd_results(args):
    """View tournament results."""
    results_dir = Path(args.path)
    summary_path = results_dir / "tournament_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        print(f"\nProtocol: {summary['protocol']}")
        print(f"Agents: {', '.join(summary['agents'])}")
        print(f"Rounds: {summary['rounds_completed']}")
        print(f"Champion: {summary['best_agent']}")
        print(f"Best fitness: {summary['best_fitness']}")
        print(f"Elapsed: {summary['elapsed_seconds']:.1f}s")
        print("\nRound history:")
        for r in summary.get("round_history", []):
            print(
                f"  Round {r['round']:3d}: "
                f"best={r['best_agent']} "
                f"fitness={r['best_fitness']:.4f} "
                f"(overall: {r['overall_best_agent']} {r['overall_best_fitness']:.4f})"
            )
    else:
        # Show JSONL results
        jsonl_path = results_dir / "results.jsonl"
        if jsonl_path.exists():
            for line in jsonl_path.read_text().splitlines():
                data = json.loads(line)
                print(
                    f"Round {data['round_num']:3d} | "
                    f"{data['agent']:10s} | "
                    f"fitness={data.get('fitness', 'N/A'):>10} | "
                    f"turns={data['turns_used']} | "
                    f"submitted={data['submitted']}"
                )
        else:
            print(f"No results found in {results_dir}")
    print()


def main():
    load_env_vars()

    parser = argparse.ArgumentParser(
        description="Crucible: Multi-agent adversarial research swarm"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # run
    run_parser = subparsers.add_parser("run", help="Run a tournament")
    run_parser.add_argument(
        "--protocol", "-p", required=True,
        help="Protocol name (kernel, quantization, interp, scaling, reward_hacking)",
    )
    run_parser.add_argument(
        "--agents", "-a", default="claude,codex",
        help="Comma-separated agent keys (default: claude,codex)",
    )
    run_parser.add_argument("--rounds", "-r", type=int, default=10, help="Number of rounds")
    run_parser.add_argument("--turns", "-t", type=int, default=15, help="Max turns per agent")
    run_parser.add_argument("--timeout", type=int, default=600, help="Timeout per agent (seconds)")
    run_parser.add_argument("--problem", help="Specific problem path (kernel protocol)")
    run_parser.add_argument("--levels", help="Comma-separated levels (kernel protocol)")
    run_parser.add_argument("--output", "-o", help="Output directory")

    # list
    subparsers.add_parser("list", help="List available protocols and agents")

    # results
    results_parser = subparsers.add_parser("results", help="View tournament results")
    results_parser.add_argument("path", help="Path to tournament output directory")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "results":
        cmd_results(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
