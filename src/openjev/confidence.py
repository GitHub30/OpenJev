"""Confidence statistics derived from an answer's probability distribution.

``probabilities`` already carries all the information; ``confidence`` collapses
the *shape* of that distribution into one number in [0, 1] so callers can
threshold without doing the math themselves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

ChoiceMethod = Literal["margin", "top1", "entropy"]
ScoreMethod = Literal["dispersion", "margin", "top1", "entropy"]


def margin(probs: Sequence[float]) -> float:
    """Gap between the best and the runner-up option (1.0 when there is a single option)."""
    if len(probs) < 2:
        return 1.0
    top = sorted(probs, reverse=True)
    return float(min(1.0, max(0.0, top[0] - top[1])))


def top1(probs: Sequence[float]) -> float:
    return float(max(probs)) if probs else 1.0


def entropy_confidence(probs: Sequence[float]) -> float:
    """1 - normalized Shannon entropy: 1 for a one-hot distribution, 0 for uniform."""
    k = len(probs)
    if k < 2:
        return 1.0
    h = -sum(p * math.log(p) for p in probs if p > 0.0)
    return float(min(1.0, max(0.0, 1.0 - h / math.log(k))))


def dispersion_confidence(probs: Sequence[float]) -> float:
    """1 - (standard deviation of the level index / the largest possible standard deviation).

    Designed for ordinal *score* answers: mass concentrated on adjacent levels is
    treated as more confident than the same mass split between the two extremes.
    """
    k = len(probs)
    if k < 2:
        return 1.0
    mean = sum(i * p for i, p in enumerate(probs))
    var = sum(((i - mean) ** 2) * p for i, p in enumerate(probs))
    max_std = (k - 1) / 2.0
    return float(min(1.0, max(0.0, 1.0 - math.sqrt(max(var, 0.0)) / max_std)))


_METHODS = {
    "margin": margin,
    "top1": top1,
    "entropy": entropy_confidence,
    "dispersion": dispersion_confidence,
}


@dataclass(frozen=True)
class ConfidenceConfig:
    choice: ChoiceMethod = "margin"
    score: ScoreMethod = "dispersion"

    def for_choice(self, probs: Sequence[float]) -> float:
        return _METHODS[self.choice](probs)

    def for_score(self, probs: Sequence[float]) -> float:
        return _METHODS[self.score](probs)
