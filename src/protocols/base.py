"""Base protocol interface for all research protocols."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from src.sandbox import Sandbox


class Protocol(ABC):
    """Abstract base for research protocols.

    A protocol defines:
    - What problem agents work on
    - How to set up the workspace
    - What system prompt and initial message to use
    - How to evaluate results
    - What fitness metric to optimize
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this protocol."""
        ...

    @property
    @abstractmethod
    def fitness_key(self) -> str:
        """Key in the metrics dict that represents fitness."""
        ...

    @property
    @abstractmethod
    def fitness_direction(self) -> str:
        """'max' to maximize fitness, 'min' to minimize."""
        ...

    @abstractmethod
    def setup_workspace(
        self, sandbox: Sandbox, round_num: int, prior_best: Optional[str]
    ) -> None:
        """Seed the sandbox workspace with problem files and context."""
        ...

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for agents."""
        ...

    @abstractmethod
    def get_initial_message(
        self, round_num: int, prior_best: Optional[str], prior_fitness: Optional[float]
    ) -> str:
        """Return the initial user message for this round."""
        ...

    @abstractmethod
    def evaluate(self, sandbox: Sandbox) -> Dict[str, Any]:
        """Run evaluation in the sandbox. Returns metrics dict.
        Must include self.fitness_key in the returned dict."""
        ...
