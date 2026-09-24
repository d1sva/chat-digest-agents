from fastapi.testclient import TestClient

from app.main import create_app
from tests.fakes import FailingLLM, ScriptedLLM

BODY = {"messages": [{"author": "Аня", "text": "Сделаю презентацию до пятницы"}]}


def make_client(llm) -> TestClient:
    return TestClient(create_app(llm=llm))


def good_llm() -> ScriptedLLM:
    return ScriptedLLM({
        "summarizer": ["Аня готовит презентацию"],
        "extractor": ['{"tasks": [{"text": "Презентация", "assignee": "Аня", "deadline": "пятница"}],'
                      ' "decisions": []}'],
        "critic": ['{"approved": true, "issues": []}'],
    })


def test_index_page():
    with make_client(good_llm()) as client:
        resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Разбор переписки" in resp.text


def test_health():
    with make_client(good_llm()) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_digest_ok():
    with make_client(good_llm()) as client:
        resp = client.post("/digest", json=BODY)
    assert resp.status_code == 200
    data = resp.json()
    assert data["summary"] == "Аня готовит презентацию"
    assert data["tasks"][0]["assignee"] == "Аня"
    assert data["rounds"] == 1
    assert len(data["trace"]) == 3


def test_empty_messages_rejected():
    with make_client(good_llm()) as client:
        resp = client.post("/digest", json={"messages": []})
    assert resp.status_code == 422


def test_llm_failure_returns_502():
    with make_client(FailingLLM()) as client:
        resp = client.post("/digest", json=BODY)
    assert resp.status_code == 502
