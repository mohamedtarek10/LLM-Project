import pytest


# --------------------------------------------------------------------------- #
# /ask validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},  # empty string
        {"question": "   "},  # whitespace only -> stripped to empty
        {"question": "\n\t"},
        {},  # missing field
        {"question": None},
        {"question": 123},
    ],
    ids=["empty", "spaces", "newlines", "missing", "null", "not-a-string"],
)
def test_ask_rejects_invalid_question_with_422(client, payload, stub_engine):
    response = client.post("/ask", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert "detail" in body
    assert any(error["loc"][-1] == "question" for error in body["detail"])
    # The engine must never be reached for an invalid request.
    assert stub_engine.calls == []


def test_ask_rejects_overlong_question_with_422(client):
    response = client.post("/ask", json={"question": "x" * 2001})
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# /ask happy path
# --------------------------------------------------------------------------- #
def test_ask_returns_answer_and_citation_sources(client, stub_engine):
    response = client.post("/ask", json={"question": "What is the delivery term?"})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"answer", "sources"}
    assert body["answer"] == "Answer to: What is the delivery term?"
    assert body["sources"] == ["Article 418"]
    assert stub_engine.calls == [("What is the delivery term?", "default")]


def test_ask_strips_surrounding_whitespace(client, stub_engine):
    client.post("/ask", json={"question": "  padded question  "})
    assert stub_engine.calls[0][0] == "padded question"


def test_ask_passes_session_id_through(client, stub_engine):
    client.post("/ask", json={"question": "hello", "session_id": "abc-123"})
    assert stub_engine.calls[0][1] == "abc-123"


def test_ask_is_503_when_engine_unavailable(client_without_engine):
    response = client_without_engine.post("/ask", json={"question": "hello"})
    assert response.status_code == 503


# --------------------------------------------------------------------------- #
# /health
# --------------------------------------------------------------------------- #
def test_health_reports_healthy_with_document_count(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "documents_indexed": 42}


def test_health_reports_unhealthy_without_engine(client_without_engine):
    response = client_without_engine.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "unhealthy", "documents_indexed": 0}


def test_validation_takes_precedence_over_unavailable_engine(client_without_engine):
    """A malformed question is a 422 even when the engine is down.

    Regression guard: resolving the engine as a `Depends(...)` ran before body
    validation and masked the 422 with a 503.
    """
    response = client_without_engine.post("/ask", json={"question": "   "})
    assert response.status_code == 422
