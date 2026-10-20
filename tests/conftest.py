import pytest
from fastapi.testclient import TestClient

from rag_service.api.app import create_app
from rag_service.rag.engine import Answer


class StubEngine:
    """Stands in for RagEngine so the API can be tested without an LLM or index."""

    def __init__(self, documents_indexed: int = 42):
        self.documents_indexed = documents_indexed
        self.calls: list[tuple[str, str]] = []

    def ask(self, question: str, session_id: str = "default") -> Answer:
        self.calls.append((question, session_id))
        return Answer(answer=f"Answer to: {question}", sources=["Article 418"])


@pytest.fixture
def stub_engine() -> StubEngine:
    return StubEngine()


@pytest.fixture
def client(stub_engine) -> TestClient:
    """A client whose app has a ready engine.

    Note: TestClient is intentionally not used as a context manager, so the
    lifespan (which would load the real FAISS index) never runs.
    """
    app = create_app()
    app.state.engine = stub_engine
    return TestClient(app)


@pytest.fixture
def client_without_engine() -> TestClient:
    app = create_app()
    app.state.engine = None
    return TestClient(app)
