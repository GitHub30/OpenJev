"""OpenJev: an open-weight System One decision engine.

Typed questions (noul / choice / score) are evaluated against a *state* in a
single forward pass over a closed candidate set, returning calibrated
probability distributions instead of generated text. The wire format matches
TypeSafe's ``POST /v1/systemone`` so the official SDKs work against it.
"""

from .backends import MockBackend, load_backend
from .calibration import Calibration
from .confidence import ConfidenceConfig
from .engine import SystemOneEngine
from .prompting import PromptConfig
from .schema import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
)

__version__ = "0.1.0"

__all__ = [
    "Calibration",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "ConfidenceConfig",
    "MockBackend",
    "NoulAnswer",
    "NoulQuestion",
    "PromptConfig",
    "ScoreAnswer",
    "ScoreQuestion",
    "SystemOneEngine",
    "SystemOneRequest",
    "SystemOneResponse",
    "Usage",
    "load_backend",
]
