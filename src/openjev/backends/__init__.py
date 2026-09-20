from .base import Backend, ScoringResult, ScoringTask
from .mock import MockBackend

__all__ = ["Backend", "ScoringResult", "ScoringTask", "MockBackend", "load_backend"]


def load_backend(spec: str, **kwargs):
    """Build a backend from a short spec string.

    ``"mock"`` gives the deterministic test backend; anything else is treated as a
    Hugging Face model id or local path and loaded with :class:`HFBackend`.
    """
    if spec == "mock":
        return MockBackend()
    from .hf import HFBackend

    return HFBackend(spec, **kwargs)
