"""Turn a System One request into scoring tasks.

A *decision* is a prompt with a closed set of candidate completions. Instead of
generating text token by token, the engine scores every candidate in a single
forward pass and converts the resulting log-probabilities into a distribution.
This is what makes the model "System One": the whole answer is a lookup over a
fixed alphabet, so it is fast, parallel across questions, and can never produce
a value outside the requested type.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from .schema import (
    ChoiceQuestion,
    JSONContent,
    NoulQuestion,
    Question,
    ScoreQuestion,
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a System One decision model. You are shown a STATE and a QUESTION "
    "about it. Answer by selecting exactly one of the listed options. Reply with "
    "the option label only. Be calibrated: when the state does not settle the "
    "question, do not pretend it does."
)


@dataclass(frozen=True)
class PromptConfig:
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    yes_label: str = "Yes"
    no_label: str = "No"
    max_state_chars: int | None = None
    """Truncate the rendered state to this many characters (None = no limit)."""


@dataclass(frozen=True)
class Prompt:
    """Messages for one decision; the backend renders them with its chat template."""

    system: str
    user: str


@dataclass
class Decision:
    """One question compiled into a prompt and a closed candidate set."""

    name: str
    kind: Literal["noul", "choice", "score"]
    prompt: Prompt
    candidates: list[str]
    """Completion strings to score, one per outcome."""
    outcomes: list[str]
    """Outcome keys aligned with ``candidates`` (option names, level indices, or true/false)."""
    legend: dict[str, JSONContent] = field(default_factory=dict)


# --------------------------------------------------------------------------- rendering


def render_content(content: JSONContent | None) -> str:
    """Render text or JSON content as a string for inclusion in a prompt."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, indent=2)


def _render_description(label: str, description: JSONContent | None) -> str:
    if description is None:
        return label
    rendered = render_content(description)
    if "\n" in rendered:
        indented = "\n".join("    " + line for line in rendered.splitlines())
        return f"{label}:\n{indented}"
    return f"{label}: {rendered}"


def render_state(state: JSONContent, config: PromptConfig) -> str:
    text = render_content(state)
    if config.max_state_chars is not None and len(text) > config.max_state_chars:
        text = text[: config.max_state_chars] + " ...[truncated]"
    return text


def _user_message(
    state_text: str,
    instructions: JSONContent | None,
    task: str,
    options: Sequence[str],
    closing: str,
) -> str:
    parts = ["## State", state_text, "", "## Question"]
    rendered = render_content(instructions).strip()
    parts.append(rendered if rendered else task)
    parts += ["", "## Options"]
    parts.extend(options)
    parts += ["", closing]
    return "\n".join(parts)


# --------------------------------------------------------------------------- compilers


def compile_noul(name: str, q: NoulQuestion, state_text: str, config: PromptConfig) -> Decision:
    yes_desc = q.criteria.true if q.criteria else None
    no_desc = q.criteria.false if q.criteria else None
    options = [
        _render_description(config.yes_label, yes_desc),
        _render_description(config.no_label, no_desc),
    ]
    user = _user_message(
        state_text,
        q.instructions,
        task="Is the statement about the state true?",
        options=options,
        closing=f"Reply with {config.yes_label} or {config.no_label} only.",
    )
    return Decision(
        name=name,
        kind="noul",
        prompt=Prompt(system=config.system_prompt, user=user),
        candidates=[config.yes_label, config.no_label],
        outcomes=["true", "false"],
    )


def compile_choice(name: str, q: ChoiceQuestion, state_text: str, config: PromptConfig) -> Decision:
    names = list(q.criteria.keys())
    labels = [str(i + 1) for i in range(len(names))]
    options = [f"{label}. {_render_description(opt, q.criteria[opt])}" for label, opt in zip(labels, names)]
    user = _user_message(
        state_text,
        q.instructions,
        task="Which option best describes the state?",
        options=options,
        closing="Reply with the option number only.",
    )
    return Decision(
        name=name,
        kind="choice",
        prompt=Prompt(system=config.system_prompt, user=user),
        candidates=labels,
        outcomes=names,
    )


def compile_score(name: str, q: ScoreQuestion, state_text: str, config: PromptConfig) -> Decision:
    labels = [str(i) for i in range(len(q.criteria))]
    options = [f"{label}. {render_content(level)}" for label, level in zip(labels, q.criteria)]
    user = _user_message(
        state_text,
        q.instructions,
        task="Which level best describes the state?",
        options=options,
        closing=(
            f"Levels are ordered from {labels[0]} (lowest) to {labels[-1]} (highest). "
            "Reply with the level number only."
        ),
    )
    return Decision(
        name=name,
        kind="score",
        prompt=Prompt(system=config.system_prompt, user=user),
        candidates=labels,
        outcomes=labels,
        legend={label: level for label, level in zip(labels, q.criteria)},
    )


def compile_question(name: str, question: Question, state_text: str, config: PromptConfig) -> Decision:
    if isinstance(question, NoulQuestion):
        return compile_noul(name, question, state_text, config)
    if isinstance(question, ChoiceQuestion):
        return compile_choice(name, question, state_text, config)
    if isinstance(question, ScoreQuestion):
        return compile_score(name, question, state_text, config)
    raise TypeError(f"unsupported question type: {type(question)!r}")


def compile_request(
    state: JSONContent,
    questions: dict[str, Any],
    config: PromptConfig | None = None,
) -> list[Decision]:
    config = config or PromptConfig()
    state_text = render_state(state, config)
    return [compile_question(name, q, state_text, config) for name, q in questions.items()]


# --------------------------------------------------------------------------- tokenization helper


def disambiguate_candidate_tokens(candidate_ids: list[list[int]], terminator: int | None) -> list[list[int]]:
    """Append a terminator when one candidate's tokens are a strict prefix of another's.

    With numeric labels "1" and "12" share a first token; without a terminator the
    shorter label would always look at least as likely as the longer one.
    """

    def is_strict_prefix(a: list[int], b: list[int]) -> bool:
        return len(a) < len(b) and b[: len(a)] == a

    ambiguous = any(is_strict_prefix(a, b) for a in candidate_ids for b in candidate_ids)
    if not ambiguous or terminator is None:
        return candidate_ids
    return [ids + [terminator] for ids in candidate_ids]
