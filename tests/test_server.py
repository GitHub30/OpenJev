import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from openjev import MockBackend, SystemOneEngine  # noqa: E402
from openjev.server import create_app  # noqa: E402


@pytest.fixture
def client():
    engine = SystemOneEngine(backend=MockBackend(), model_name="openjev-test")
    return TestClient(create_app(engine))


def test_systemone_endpoint(client):
    body = {
        "state": "Hi, I've been trying to connect my Stripe account for 3 days and it keeps failing. I'm losing sales. Please help ASAP.",
        "model": "jev-latest",
        "questions": {
            "urgency": {"type": "noul", "instructions": "Does this message express urgency?"},
            "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "charges, invoices", "integrations": "connect Stripe account failing", "other": None}},
        },
    }
    r = client.post("/v1/systemone", json=body, headers={"Authorization": "Bearer anything"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["model"] == "openjev-test"
    assert data["answers"]["urgency"]["type"] == "noul"
    assert data["answers"]["team"]["choice"] == "integrations"
    assert set(data["usage"]) == {"input_tokens", "output_tokens"}


def test_validation_error_shape(client):
    r = client.post("/v1/systemone", json={"state": "x", "model": "jev-latest", "questions": {"q": {"type": "score"}}})
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"][:3] == ["body", "questions", "q"]


def test_unknown_model(client):
    r = client.post("/v1/systemone", json={"state": "x", "model": "gpt-9", "questions": {"q": {"type": "noul"}}})
    assert r.status_code == 404


def test_models_listing(client):
    r = client.get("/v1/models")
    names = {m["name"] for m in r.json()["models"]}
    assert {"openjev-test", "jev-latest", "jev-preview"} <= names


def test_api_key_enforced(monkeypatch):
    monkeypatch.setenv("OPENJEV_API_KEY", "secret")
    engine = SystemOneEngine(backend=MockBackend())
    c = TestClient(create_app(engine))
    body = {"state": "x", "model": "jev-latest", "questions": {"q": {"type": "noul"}}}
    assert c.post("/v1/systemone", json=body).status_code == 401
    assert c.post("/v1/systemone", json=body, headers={"Authorization": "Bearer secret"}).status_code == 200
