"""Backend protocol: score closed candidate sets against prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from ..prompting import Prompt


@dataclass(frozen=True)
class ScoringTask:
    prompt: Prompt
    candidates: list[str]
    """Completion strings; the backend returns one log-probability per candidate."""


@dataclass(frozen=True)
class ScoringResult:
    logprobs: list[float]
    """Unnormalized log P(candidate | prompt), one per candidate (same order as the task)."""
    prompt_tokens: int
    candidate_tokens: int


@runtime_checkable
class Backend(Protocol):
    """Anything that can score candidate completions.

    Implementations must evaluate *all* tasks in a request together when they can;
    the engine hands over the whole request in one call precisely so that a
    backend can batch every decision into a single forward pass.
    """

    name: str

    def score(self, tasks: Sequence[ScoringTask]) -> list[ScoringResult]: ...
