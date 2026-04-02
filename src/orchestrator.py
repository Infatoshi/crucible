"""Tournament orchestrator for multi-agent adversarial research."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.agent import run_agent
from src.models import ModelConfig, get_model_config
from src.protocols.base import Protocol
from src.run_log import AgentResult, RunLog
from src.sandbox import Sandbox, SandboxConfig


class Tournament:
    """Runs N agents in a tournament: each round, all agents compete,
    best solution seeds the next round."""

    def __init__(
        self,
        protocol: Protocol,
        agent_keys: List[str],
        max_rounds: int = 10,
        max_turns_per_agent: int = 15,
        timeout_per_agent: int = 600,
        output_dir: Optional[Path] = None,
    ):
        self.protocol = protocol
        self.agent_keys = agent_keys
        self.agent_configs: Dict[str, ModelConfig] = {
            key: get_model_config(key) for key in agent_keys
        }
        self.max_rounds = max_rounds
        self.max_turns = max_turns_per_agent
        self.timeout = timeout_per_agent

        if output_dir is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("outputs") / f"tournament_{self.protocol.name}_{ts}"
        self.output_dir = output_dir
        self.run_log = RunLog(output_dir)

        self.best_solution: Optional[str] = None
        self.best_fitness: Optional[float] = None
        self.best_agent: Optional[str] = None
        self.round_history: List[dict] = []

    def run(self) -> dict:
        """Run the full tournament. Returns summary dict."""
        print(f"\n{'='*60}", flush=True)
        print("CRUCIBLE TOURNAMENT", flush=True)
        print(f"Protocol: {self.protocol.name}", flush=True)
        print(f"Agents: {', '.join(self.agent_keys)}", flush=True)
        print(f"Rounds: {self.max_rounds}", flush=True)
        print(f"Output: {self.output_dir}", flush=True)
        print(f"{'='*60}\n", flush=True)

        t0 = time.time()

        for round_num in range(1, self.max_rounds + 1):
            print(f"\n{'─'*60}", flush=True)
            print(f"ROUND {round_num}/{self.max_rounds}", flush=True)
            if self.best_agent:
                print(
                    f"Defending champion: {self.best_agent} "
                    f"(fitness={self.best_fitness:.4f})",
                    flush=True,
                )
            print(f"{'─'*60}", flush=True)

            round_results = self._run_round(round_num)
            best_this_round = self._select_best(round_results)

            if best_this_round and best_this_round.fitness is not None:
                improved = False
                if self.best_fitness is None:
                    improved = True
                elif self.protocol.fitness_direction == "max":
                    improved = best_this_round.fitness > self.best_fitness
                else:
                    improved = best_this_round.fitness < self.best_fitness

                if improved:
                    self.best_fitness = best_this_round.fitness
                    self.best_solution = best_this_round.solution_code
                    self.best_agent = best_this_round.agent
                    print(
                        f"\n  NEW CHAMPION: {self.best_agent} "
                        f"fitness={self.best_fitness:.4f}",
                        flush=True,
                    )
                else:
                    print(
                        f"\n  No improvement. Champion remains: {self.best_agent} "
                        f"fitness={self.best_fitness:.4f}",
                        flush=True,
                    )

            # Log round summary
            self.run_log.write_round_summary(round_num, round_results, best_this_round)
            self.round_history.append({
                "round": round_num,
                "best_agent": best_this_round.agent if best_this_round else None,
                "best_fitness": best_this_round.fitness if best_this_round else None,
                "overall_best_fitness": self.best_fitness,
                "overall_best_agent": self.best_agent,
            })

        elapsed = time.time() - t0
        summary = {
            "protocol": self.protocol.name,
            "agents": self.agent_keys,
            "rounds_completed": len(self.round_history),
            "best_agent": self.best_agent,
            "best_fitness": self.best_fitness,
            "elapsed_seconds": elapsed,
            "round_history": self.round_history,
        }

        # Write final summary
        import json
        (self.output_dir / "tournament_summary.json").write_text(json.dumps(summary, indent=2))

        print(f"\n{'='*60}", flush=True)
        print("TOURNAMENT COMPLETE", flush=True)
        print(f"Champion: {self.best_agent}", flush=True)
        print(f"Best fitness: {self.best_fitness}", flush=True)
        print(f"Elapsed: {elapsed:.1f}s", flush=True)
        print(f"Results: {self.output_dir}", flush=True)
        print(f"{'='*60}\n", flush=True)

        return summary

    def _run_round(self, round_num: int) -> List[AgentResult]:
        """Run all agents for a single round."""
        results = []
        for agent_key in self.agent_keys:
            model_config = self.agent_configs[agent_key]
            print(f"\n  Agent: {agent_key} ({model_config.name})", flush=True)

            # Create fresh sandbox for this agent
            sandbox = Sandbox(SandboxConfig(timeout=self.timeout))
            sandbox.start()
            # Init git repo (Codex requires it)
            sandbox.run_command("git init -q && git add -A && git commit -q -m init --allow-empty", timeout=10)

            try:
                # Let protocol set up the workspace
                self.protocol.setup_workspace(
                    sandbox, round_num, self.best_solution
                )

                # Get prompts from protocol
                system_prompt = self.protocol.get_system_prompt()
                initial_message = self.protocol.get_initial_message(
                    round_num, self.best_solution, self.best_fitness
                )

                # Run the agent
                result = run_agent(
                    model_config=model_config,
                    sandbox=sandbox,
                    system_prompt=system_prompt,
                    initial_message=initial_message,
                    max_turns=self.max_turns,
                    timeout_seconds=self.timeout,
                    run_log=self.run_log,
                    agent_name=agent_key,
                    round_num=round_num,
                    protocol_name=self.protocol.name,
                )

                # Evaluate the result
                if result.submitted and result.solution_code:
                    try:
                        metrics = self.protocol.evaluate(sandbox)
                        result.metrics = metrics
                        result.fitness = metrics.get(self.protocol.fitness_key)
                        print(
                            f"  [{agent_key}] Evaluation: {metrics}",
                            flush=True,
                        )
                    except Exception as e:
                        result.error = f"Evaluation failed: {e}"
                        print(f"  [{agent_key}] Eval error: {e}", flush=True)
                else:
                    print(f"  [{agent_key}] No submission", flush=True)

                self.run_log.log_result(result)
                results.append(result)

            finally:
                sandbox.stop()

        return results

    def _select_best(self, results: List[AgentResult]) -> Optional[AgentResult]:
        """Select the best result from this round."""
        valid = [r for r in results if r.fitness is not None]
        if not valid:
            return None
        if self.protocol.fitness_direction == "max":
            return max(valid, key=lambda r: r.fitness)
        else:
            return min(valid, key=lambda r: r.fitness)
