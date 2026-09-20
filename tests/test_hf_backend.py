"""Opt-in integration test for the transformers backend.

    OPENJEV_TEST_MODEL=HuggingFaceTB/SmolLM2-135M-Instruct pytest tests/test_hf_backend.py
"""

import math
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

MODEL = os.environ.get("OPENJEV_TEST_MODEL")
pytestmark = pytest.mark.skipif(not MODEL, reason="set OPENJEV_TEST_MODEL to run")


@pytest.fixture(scope="module")
def backend():
    from openjev.backends.hf import HFBackend

    return HFBackend(MODEL, device="cpu", dtype="float32", max_batch_tokens=2048)


@pytest.fixture
def request_body():
    from openjev import SystemOneRequest

    return SystemOneRequest.model_validate({
        "state": "Hi, I've been trying to connect my Stripe account for 3 days and it keeps failing. Please help ASAP.",
        "questions": {
            "urgency": {"type": "noul", "instructions": "Does this message express urgency?"},
            "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": None, "integrations": None, "other": None}},
            "severity": {"type": "score", "instructions": "How severe?", "criteria": ["cosmetic", "workaround exists", "blocking"]},
            "wide": {"type": "choice", "instructions": "Pick one.", "criteria": {f"opt{i}": None for i in range(12)}},
        },
    })


def test_shared_prefix_matches_full_sequences(backend, request_body):
    from openjev import SystemOneEngine

    engine = SystemOneEngine(backend=backend)
    backend.share_prefix = True
    shared, usage = engine.evaluate_raw(request_body)
    backend.share_prefix = False
    full, _ = engine.evaluate_raw(request_body)
    for a, b in zip(shared, full):
        assert a.decision.name == b.decision.name
        for x, y in zip(a.logprobs, b.logprobs):
            assert abs(x - y) < 1e-3
    assert usage.input_tokens > 0
    # 12 numeric labels include "1" vs "10".."12", so a terminator is appended to each candidate.
    wide = next(r for r in shared if r.decision.name == "wide")
    assert len(wide.probabilities) == 12
    assert math.isclose(sum(wide.probabilities), 1.0, abs_tol=1e-6)


def test_answers_are_well_formed(backend, request_body):
    from openjev import SystemOneEngine

    response = SystemOneEngine(backend=backend).evaluate(request_body)
    assert 0.0 <= response.answers["urgency"].noul <= 1.0
    assert response.answers["team"].choice in {"billing", "integrations", "other"}
    assert 0.0 <= response.answers["severity"].score <= 2.0
