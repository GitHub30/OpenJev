"""The System One engine: request in, typed answers out, no generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .backends.base import Backend, ScoringResult, ScoringTask
from .calibration import Calibration, softmax
from .confidence import ConfidenceConfig
from .prompting import Decision, PromptConfig, compile_request
from .schema import (
    Answer,
    ChoiceAnswer,
    JSONContent,
    NoulAnswer,
    ScoreAnswer,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
)


@dataclass
class RawDecision:
    """A decision together with its raw log-probabilities (useful for calibration and evals)."""

    decision: Decision
    logprobs: list[float]
    probabilities: list[float]


@dataclass
class SystemOneEngine:
    backend: Backend
    model_name: str = "openjev"
    prompt_config: PromptConfig = field(default_factory=PromptConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    calibration: Calibration = field(default_factory=Calibration)

    # ------------------------------------------------------------------ public API

    def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        raws, usage = self.evaluate_raw(request)
        answers: dict[str, Answer] = {raw.decision.name: self._to_answer(raw) for raw in raws}
        return SystemOneResponse(model=self.model_name, answers=answers, usage=usage)

    def system_one(
        self,
        state: JSONContent,
        questions: Mapping[str, Any],
        model: str | None = None,
    ) -> SystemOneResponse:
        """Convenience wrapper mirroring the official SDK's ``client.system_one``."""
        payload = {"state": state, "questions": dict(questions)}
        if model is not None:
            payload["model"] = model
        return self.evaluate(SystemOneRequest.model_validate(payload))

    def evaluate_raw(self, request: SystemOneRequest) -> tuple[list[RawDecision], Usage]:
        decisions = compile_request(request.state, request.questions, self.prompt_config)
        tasks = [ScoringTask(prompt=d.prompt, candidates=d.candidates) for d in decisions]
        results: list[ScoringResult] = self.backend.score(tasks) if tasks else []
        if len(results) != len(tasks):
            raise RuntimeError(f"backend returned {len(results)} results for {len(tasks)} tasks")

        raws: list[RawDecision] = []
        for decision, result in zip(decisions, results):
            temperature = self.calibration.temperature(decision.kind)
            probs = softmax(result.logprobs, temperature)
            raws.append(RawDecision(decision=decision, logprobs=list(result.logprobs), probabilities=probs))

        usage = Usage(
            input_tokens=sum(r.prompt_tokens for r in results),
            output_tokens=sum(r.candidate_tokens for r in results),
        )
        return raws, usage

    # ------------------------------------------------------------------ answers

    def _to_answer(self, raw: RawDecision) -> Answer:
        d, probs = raw.decision, raw.probabilities
        if d.kind == "noul":
            return NoulAnswer(noul=_clamp(probs[0]))
        if d.kind == "choice":
            best = max(range(len(probs)), key=probs.__getitem__)
            return ChoiceAnswer(
                choice=d.outcomes[best],
                confidence=_clamp(self.confidence.for_choice(probs)),
                probabilities={name: p for name, p in zip(d.outcomes, probs)},
            )
        if d.kind == "score":
            score = sum(i * p for i, p in enumerate(probs))
            return ScoreAnswer(
                score=score,
                confidence=_clamp(self.confidence.for_score(probs)),
                legend=dict(d.legend),
                probabilities={label: p for label, p in zip(d.outcomes, probs)},
            )
        raise ValueError(f"unknown decision kind {d.kind!r}")


def _clamp(x: float) -> float:
    return float(min(1.0, max(0.0, x)))
