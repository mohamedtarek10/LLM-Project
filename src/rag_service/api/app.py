"""FastAPI application exposing the RAG pipeline."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status

from rag_service import __version__
from rag_service.api.dependencies import require_engine
from rag_service.api.schemas import AskRequest, AskResponse, HealthResponse
from rag_service.rag import RagEngine, get_engine

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the index once at startup.

    A failure here is not fatal: the app still starts so that /health can
    report the problem instead of the container dying in a crash loop.
    """
    try:
        app.state.engine = get_engine()
        logger.info("RAG engine ready: %s documents indexed", app.state.engine.documents_indexed)
    except Exception:
        app.state.engine = None
        logger.exception("Failed to initialise the RAG engine")
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Q&A Service",
        version=__version__,
        description="Ask questions over the indexed corpus; answers cite their sources.",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health() -> HealthResponse:
        engine: RagEngine | None = getattr(app.state, "engine", None)
        if engine is None:
            return HealthResponse(status="unhealthy", documents_indexed=0)
        return HealthResponse(status="healthy", documents_indexed=engine.documents_indexed)

    @app.post("/ask", response_model=AskResponse, tags=["qa"])
    def ask(payload: AskRequest) -> AskResponse:
        # Reaching this line means Pydantic already accepted the question, so a
        # malformed request is always a 422 and never a 503.
        engine = require_engine(app)
        try:
            result = engine.ask(payload.question, session_id=payload.session_id)
        except Exception as exc:
            logger.exception("Failed to answer question")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to generate an answer: {exc}",
            ) from exc
        return AskResponse(answer=result.answer, sources=result.sources)

    return app


app = create_app()
