"""JSONL logging and turn artifact management."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AgentResult:
    agent: str
    model_id: str
    round_num: int
    protocol: str
    fitness: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    turns_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    submitted: bool = False
    error: Optional[str] = None
    solution_code: Optional[str] = None
    elapsed_seconds: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class RunLog:
    """Manages a single run's logging directory."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = output_dir / "results.jsonl"
        self.turns_dir = output_dir / "turns"
        self.turns_dir.mkdir(exist_ok=True)

    def log_result(self, result: AgentResult):
        with open(self.results_path, "a") as f:
            f.write(json.dumps(result.to_dict()) + "\n")

    def get_turn_dir(self, agent: str, round_num: int) -> Path:
        d = self.turns_dir / f"{agent}_round_{round_num:03d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_turn_artifact(self, agent: str, round_num: int, turn: int, suffix: str, content: str):
        turn_dir = self.get_turn_dir(agent, round_num)
        path = turn_dir / f"turn_{turn:02d}_{suffix}"
        path.write_text(content)

    def load_results(self) -> List[AgentResult]:
        results = []
        if not self.results_path.exists():
            return results
        for line in self.results_path.read_text().splitlines():
            if line.strip():
                data = json.loads(line)
                results.append(AgentResult(**data))
        return results

    def write_round_summary(self, round_num: int, results: List[AgentResult], best: Optional[AgentResult]):
        summary_path = self.output_dir / f"round_{round_num:03d}_summary.json"
        summary = {
            "round": round_num,
            "results": [r.to_dict() for r in results],
            "best": best.to_dict() if best else None,
        }
        summary_path.write_text(json.dumps(summary, indent=2))
