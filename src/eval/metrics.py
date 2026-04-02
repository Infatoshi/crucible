"""Metric computation and aggregation utilities."""

from __future__ import annotations

import math
from typing import Any, Dict, List

from src.run_log import AgentResult


def geometric_mean(values: List[float]) -> float:
    """Compute geometric mean of positive values."""
    positive = [v for v in values if v > 0]
    if not positive:
        return 0.0
    log_sum = sum(math.log(v) for v in positive)
    return math.exp(log_sum / len(positive))


def summarize_tournament(results: List[AgentResult]) -> Dict[str, Any]:
    """Compute aggregate statistics for a tournament."""
    by_agent: Dict[str, List[AgentResult]] = {}
    for r in results:
        by_agent.setdefault(r.agent, []).append(r)

    agent_stats = {}
    for agent, agent_results in by_agent.items():
        fitnesses = [r.fitness for r in agent_results if r.fitness is not None]
        agent_stats[agent] = {
            "rounds_played": len(agent_results),
            "submissions": sum(1 for r in agent_results if r.submitted),
            "errors": sum(1 for r in agent_results if r.error),
            "mean_fitness": sum(fitnesses) / len(fitnesses) if fitnesses else None,
            "max_fitness": max(fitnesses) if fitnesses else None,
            "min_fitness": min(fitnesses) if fitnesses else None,
            "geo_mean_fitness": geometric_mean(fitnesses) if fitnesses else None,
            "total_input_tokens": sum(r.input_tokens for r in agent_results),
            "total_output_tokens": sum(r.output_tokens for r in agent_results),
            "mean_turns": sum(r.turns_used for r in agent_results) / len(agent_results),
        }

    return {
        "n_results": len(results),
        "n_agents": len(by_agent),
        "agent_stats": agent_stats,
    }
