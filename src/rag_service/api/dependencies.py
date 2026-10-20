"""Engine lookup, isolated so tests can substitute a stub."""

from fastapi import FastAPI, HTTPException, status

from rag_service.rag import RagEngine


def require_engine(app: FastAPI) -> RagEngine:
    """Return the initialised engine, or fail with 503.

    Called from inside the handler body rather than as a `Depends(...)`:
    FastAPI resolves dependencies *before* validating the request body, so a
    raising dependency would mask a malformed question's 422 with a 503.
    """
    engine = getattr(app.state, "engine", None)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG engine is not initialised; check the index and credentials.",
        )
    return engine
