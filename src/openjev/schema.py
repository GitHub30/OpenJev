"""Wire schema for the System One API.

These models mirror the request/response shapes of ``POST /v1/systemone`` as
published in TypeSafe's OpenAPI document, so that the official ``typesafe-sdk``
clients can talk to an OpenJev server unchanged.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

JSONContent: TypeAlias = str | dict[str, Any] | list[Any]
"""Text, a JSON object, or an array. Used for state, instructions and criteria."""

MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- questions


class NoulCriteria(_Strict):
    true: JSONContent | None = Field(None, description="What counts as a yes answer.")
    false: JSONContent | None = Field(None, description="What counts as a no answer.")


class NoulQuestion(_Strict):
    type: Literal["noul"]
    instructions: JSONContent | None = Field(None, description="The yes/no question or statement to evaluate.")
    criteria: NoulCriteria | None = None


class ChoiceQuestion(_Strict):
    type: Literal["choice"]
    instructions: JSONContent | None = Field(None, description="What the model should decide when choosing an option.")
    criteria: dict[str, JSONContent | None] = Field(
        ..., description="Choice names mapped to descriptions of when each applies."
    )

    @model_validator(mode="after")
    def _check_options(self) -> ChoiceQuestion:
        if not self.criteria:
            raise ValueError("choice criteria must contain at least one option")
        if len(self.criteria) > MAX_CHOICE_OPTIONS:
            raise ValueError(f"choice criteria supports at most {MAX_CHOICE_OPTIONS} options")
        return self


class ScoreQuestion(_Strict):
    type: Literal["score"]
    instructions: JSONContent | None = Field(None, description="What the model should rate.")
    criteria: list[JSONContent] = Field(
        ..., min_length=1, max_length=MAX_SCORE_LEVELS,
        description="Ordered level descriptions; position is the score, starting at zero.",
    )


Question: TypeAlias = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class SystemOneRequest(_Strict):
    state: JSONContent = Field(..., description="The content all questions in this request refer to.")
    model: str = Field("jev-latest", description="Model name or alias.")
    questions: dict[str, Question] = Field(..., min_length=1)


# --------------------------------------------------------------------------- answers


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float = Field(..., ge=0.0, le=1.0, description="Probability that the answer is yes.")


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[str, float]


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    confidence: float = Field(..., ge=0.0, le=1.0)
    legend: dict[str, JSONContent]
    probabilities: dict[str, float]


Answer: TypeAlias = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage


class ModelMetadata(BaseModel):
    name: str
    description: str
    release_date: str


class ModelMetadataList(BaseModel):
    models: list[ModelMetadata]
