"""Post-hoc calibration of the candidate distributions.

A System One answer is only useful if its probabilities are *epistemically
honest*: a noul of 0.8 should be right about 80% of the time. The cheapest
reliable fix for an off-the-shelf LLM is temperature scaling, fitted per
question type on a small labelled set. It never changes which option wins,
only how sharp the distribution is.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Literal, Sequence

from pydantic import BaseModel, Field

Kind = Literal["noul", "choice", "score"]


def softmax(logprobs: Sequence[float], temperature: float = 1.0) -> list[float]:
    scaled = [lp / temperature for lp in logprobs]
    m = max(scaled)
    exps = [math.exp(v - m) for v in scaled]
    z = sum(exps)
    return [e / z for e in exps]


class Calibration(BaseModel):
    """Per-question-type temperatures (1.0 = the raw model distribution)."""

    temperatures: dict[str, float] = Field(default_factory=lambda: {"noul": 1.0, "choice": 1.0, "score": 1.0})

    def temperature(self, kind: Kind) -> float:
        return self.temperatures.get(kind, 1.0)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def nll(logprob_sets: Sequence[Sequence[float]], gold: Sequence[int], temperature: float) -> float:
    total = 0.0
    for lps, g in zip(logprob_sets, gold):
        probs = softmax(lps, temperature)
        total -= math.log(max(probs[g], 1e-12))
    return total / max(len(gold), 1)


def fit_temperature(
    logprob_sets: Sequence[Sequence[float]],
    gold: Sequence[int],
    *,
    lo: float = 0.05,
    hi: float = 20.0,
    iters: int = 60,
) -> float:
    """Golden-section search for the temperature minimising NLL on log-scale."""
    if not logprob_sets:
        return 1.0
    a, b = math.log(lo), math.log(hi)
    phi = (math.sqrt(5) - 1) / 2
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = nll(logprob_sets, gold, math.exp(c))
    fd = nll(logprob_sets, gold, math.exp(d))
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = nll(logprob_sets, gold, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = nll(logprob_sets, gold, math.exp(d))
    return math.exp((a + b) / 2)


def expected_calibration_error(confidences: Iterable[float], correct: Iterable[bool], bins: int = 10) -> float:
    """Standard ECE over equal-width confidence bins."""
    conf = list(confidences)
    corr = list(correct)
    if not conf:
        return 0.0
    totals = [0] * bins
    sum_conf = [0.0] * bins
    sum_acc = [0.0] * bins
    for c, ok in zip(conf, corr):
        i = min(int(c * bins), bins - 1)
        totals[i] += 1
        sum_conf[i] += c
        sum_acc[i] += 1.0 if ok else 0.0
    n = len(conf)
    ece = 0.0
    for i in range(bins):
        if totals[i]:
            ece += (totals[i] / n) * abs(sum_acc[i] / totals[i] - sum_conf[i] / totals[i])
    return ece
