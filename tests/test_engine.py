import math

import pytest

from openjev import MockBackend, SystemOneEngine, SystemOneRequest
from openjev.backends.base import ScoringResult, ScoringTask
from openjev.calibration import Calibration, fit_temperature, softmax
from openjev.confidence import dispersion_confidence, entropy_confidence, margin
from openjev.prompting import compile_request, disambiguate_candidate_tokens


@pytest.fixture
def engine():
    return SystemOneEngine(backend=MockBackend(), model_name="openjev-test")


def test_mixed_questions_round_trip(engine):
    response = engine.system_one(
        "I was charged twice for my Stripe subscription. Please fix this ASAP, I am losing sales.",
        {
            "billing": {"type": "noul", "instructions": "Is this about billing?", "criteria": {"true": "charged twice, subscription, invoice"}},
            "tone": {"type": "choice", "instructions": "What is the tone?", "criteria": {"calm": "relaxed", "angry": "losing sales ASAP"}},
            "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["can wait", "this week", "ASAP losing sales"]},
        },
    )
    assert response.model == "openjev-test"
    assert set(response.answers) == {"billing", "tone", "urgency"}

    noul = response.answers["billing"]
    assert noul.type == "noul" and noul.noul > 0.5

    choice = response.answers["tone"]
    assert choice.type == "choice"
    assert choice.choice == "angry"
    assert math.isclose(sum(choice.probabilities.values()), 1.0, abs_tol=1e-9)
    assert set(choice.probabilities) == {"calm", "angry"}
    assert 0.0 <= choice.confidence <= 1.0

    score = response.answers["urgency"]
    assert score.type == "score"
    assert score.legend == {"0": "can wait", "1": "this week", "2": "ASAP losing sales"}
    assert set(score.probabilities) == {"0", "1", "2"}
    assert 0.0 <= score.score <= 2.0
    assert score.score > 1.0

    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens == 2 + 2 + 3


def test_wire_format_matches_sdk_expectations(engine):
    payload = {
        "state": {"subject": "Duplicate charge", "message": "Please help."},
        "model": "jev-latest",
        "questions": {"billing": {"type": "noul", "instructions": "Is this message about billing?"}},
    }
    response = engine.evaluate(SystemOneRequest.model_validate(payload))
    dumped = response.model_dump()
    assert set(dumped) == {"model", "answers", "usage"}
    assert set(dumped["usage"]) == {"input_tokens", "output_tokens"}
    assert set(dumped["answers"]["billing"]) == {"type", "noul"}


def test_choice_limits():
    ok = {"type": "choice", "criteria": {str(i): None for i in range(255)}}
    SystemOneRequest.model_validate({"state": "x", "questions": {"q": ok}})
    too_many = {"type": "choice", "criteria": {str(i): None for i in range(256)}}
    with pytest.raises(Exception):
        SystemOneRequest.model_validate({"state": "x", "questions": {"q": too_many}})
    with pytest.raises(Exception):
        SystemOneRequest.model_validate({"state": "x", "questions": {"q": {"type": "score", "criteria": []}}})
    with pytest.raises(Exception):
        SystemOneRequest.model_validate({"state": "x", "questions": {}})


def test_prompt_compilation_uses_numeric_labels():
    decisions = compile_request(
        "hello",
        SystemOneRequest.model_validate(
            {"state": "hello", "questions": {
                "c": {"type": "choice", "criteria": {"a": "alpha", "b": None}},
                "s": {"type": "score", "criteria": ["low", "high"]},
                "n": {"type": "noul", "instructions": "Is it a greeting?"},
            }}
        ).questions,
    )
    by_name = {d.name: d for d in decisions}
    assert by_name["c"].candidates == ["1", "2"] and by_name["c"].outcomes == ["a", "b"]
    assert "1. a: alpha" in by_name["c"].prompt.user and "2. b" in by_name["c"].prompt.user
    assert by_name["s"].candidates == ["0", "1"]
    assert by_name["n"].candidates == ["Yes", "No"] and by_name["n"].outcomes == ["true", "false"]
    # All decisions in a request share the same state prefix (enables KV-cache sharing).
    prefixes = {d.prompt.user.split("## Question")[0] for d in decisions}
    assert len(prefixes) == 1


def test_disambiguation_appends_terminator_only_when_needed():
    assert disambiguate_candidate_tokens([[1], [2]], terminator=99) == [[1], [2]]
    assert disambiguate_candidate_tokens([[1], [1, 2]], terminator=99) == [[1, 99], [1, 2, 99]]
    assert disambiguate_candidate_tokens([[1], [1, 2]], terminator=None) == [[1], [1, 2]]


def test_confidence_statistics():
    assert margin([0.9, 0.1]) == pytest.approx(0.8)
    assert margin([1.0]) == 1.0
    assert entropy_confidence([0.5, 0.5]) == pytest.approx(0.0)
    assert entropy_confidence([1.0, 0.0]) == pytest.approx(1.0)
    # Adjacent mass is more confident than mass on the extremes for ordinal scores.
    assert dispersion_confidence([0.5, 0.5, 0.0]) > dispersion_confidence([0.5, 0.0, 0.5])
    assert dispersion_confidence([0.0, 1.0, 0.0]) == pytest.approx(1.0)


def test_temperature_scaling_sharpens_or_softens():
    lps = [math.log(0.6), math.log(0.4)]
    assert softmax(lps, 1.0)[0] == pytest.approx(0.6)
    assert softmax(lps, 0.5)[0] > 0.6
    assert softmax(lps, 2.0)[0] < 0.6

    # An overconfident model (always ~0.9 but right 60% of the time) should get T > 1.
    sets = [[math.log(0.9), math.log(0.1)]] * 10
    gold = [0] * 6 + [1] * 4
    assert fit_temperature(sets, gold) > 1.0
    # A well-calibrated model stays near T = 1.
    sets = [[math.log(0.9), math.log(0.1)]] * 10
    gold = [0] * 9 + [1]
    assert fit_temperature(sets, gold) == pytest.approx(1.0, abs=0.15)


def test_calibration_applied_in_engine():
    class Fixed:
        name = "fixed"

        def score(self, tasks):
            return [ScoringResult(logprobs=[math.log(0.6), math.log(0.4)], prompt_tokens=1, candidate_tokens=2) for _ in tasks]

    hot = SystemOneEngine(backend=Fixed(), calibration=Calibration(temperatures={"noul": 0.5}))
    cold = SystemOneEngine(backend=Fixed(), calibration=Calibration(temperatures={"noul": 2.0}))
    q = {"n": {"type": "noul", "instructions": "?"}}
    assert hot.system_one("s", q).answers["n"].noul > 0.6
    assert cold.system_one("s", q).answers["n"].noul < 0.6


def test_backend_receives_all_tasks_in_one_call():
    calls: list[int] = []

    class Spy:
        name = "spy"

        def score(self, tasks):
            calls.append(len(tasks))
            return [ScoringResult(logprobs=[0.0] * len(t.candidates), prompt_tokens=1, candidate_tokens=len(t.candidates)) for t in tasks]

    engine = SystemOneEngine(backend=Spy())
    engine.system_one("s", {f"q{i}": {"type": "noul", "instructions": "?"} for i in range(7)})
    assert calls == [7]
