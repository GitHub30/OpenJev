"""Deterministic backend for tests and for exercising the server without a model.

Scores are a lexical-overlap heuristic between the state and each option's text,
so obviously matching inputs produce sensible answers and everything is
reproducible. It is not intended to be smart.
"""

from __future__ import annotations

import math
import re
from typing import Sequence

from ..prompting import Prompt
from .base import ScoringResult, ScoringTask

_WORD = re.compile(r"[A-Za-z0-9']+")


def _tokens(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text)}


def _section(user: str, header: str) -> str:
    marker = f"## {header}\n"
    start = user.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    end = user.find("\n## ", start)
    return user[start:] if end < 0 else user[start:end]


def _option_lines(user: str) -> list[str]:
    lines = [ln for ln in _section(user, "Options").splitlines() if ln.strip()]
    return lines


class MockBackend:
    name = "mock"

    def __init__(self, sharpness: float = 2.0) -> None:
        self.sharpness = sharpness

    def _score_prompt(self, prompt: Prompt, candidates: list[str]) -> list[float]:
        state_words = _tokens(_section(prompt.user, "State"))
        question_words = _tokens(_section(prompt.user, "Question"))
        option_lines = _option_lines(prompt.user)
        scores: list[float] = []
        for i, label in enumerate(candidates):
            line = option_lines[i] if i < len(option_lines) else label
            # Strip the "1. " / "Yes: " label prefix so we only compare the description.
            description = re.sub(r"^\s*[^:.\s]+[.:]\s*", "", line) or line
            words = _tokens(description) - {label.lower()}
            overlap = len(words & state_words) + 0.5 * len(words & question_words)
            scores.append(self.sharpness * overlap)
        # Deterministic tie-break so the answer never depends on dict ordering luck.
        return [s - 1e-3 * i for i, s in enumerate(scores)]

    def score(self, tasks: Sequence[ScoringTask]) -> list[ScoringResult]:
        results: list[ScoringResult] = []
        for task in tasks:
            raw = self._score_prompt(task.prompt, task.candidates)
            norm = math.log(sum(math.exp(s) for s in raw))
            logprobs = [s - norm for s in raw]
            prompt_tokens = len(task.prompt.system.split()) + len(task.prompt.user.split())
            results.append(ScoringResult(logprobs=logprobs, prompt_tokens=prompt_tokens, candidate_tokens=len(task.candidates)))
        return results
