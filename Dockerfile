# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# --- dependencies (cached independently of the source) ---------------------- #
COPY pyproject.toml README.md ./
COPY src/rag_service/__init__.py src/rag_service/__init__.py
RUN pip install --upgrade pip && pip install -e .

# --- application ------------------------------------------------------------ #
COPY src/ src/

# --- the corpus and the prebuilt index ship inside the image ---------------- #
# The service answers questions with no external data dependency at run time.
COPY data/ data/
COPY files/ files/
COPY faiss_ev/ faiss_ev/

ENV RAG_INDEX_DIR=/app/faiss_ev \
    RAG_CORPUS_JSON=/app/data/corpus.json \
    RAG_SOURCE_DIR=/app/files

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"

CMD ["uvicorn", "rag_service.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
