# LLM-Project — RAG Q&A Service

Retrieval-augmented question answering over a structured document corpus.
Answers cite their sources as **article citations**, not chunk ids.

## Run the Q&A service (3 commands)

```bash
docker build -t rag-service .
docker run -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY rag-service
curl -s localhost:8000/health
```

The image ships with the corpus and the prebuilt vector store, so no data
download or index build is needed on the target machine. Only
`OPENAI_API_KEY` is required at run time (to embed the incoming question and
call the generative model).

### Ask a question

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' -d '{"question": "What are the obligations of the seller?"}'
```

```json
{
  "answer": "...",
  "sources": ["Article 418", "Article 419 (repealed)"]
}
```

Interactive docs: <http://localhost:8000/docs>

## API

| Method | Path      | Request                                      | Response |
| ------ | --------- | -------------------------------------------- | -------- |
| `GET`  | `/health` | –                                            | `{"status": "healthy", "documents_indexed": 1234}` |
| `POST` | `/ask`    | `{"question": str, "session_id": str?}`      | `{"answer": str, "sources": [str]}` |

An empty or whitespace-only `question` is rejected with **422** before the
model is called (see `tests/test_api.py`).

## Local development

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,ingest,ui]"
pytest
```

Run the API locally:

```bash
uvicorn rag_service.api.app:app --reload --port 8000
```

Run the Streamlit UI (same engine, same citations):

```bash
streamlit run streamlit_app.py
```

## Rebuilding the index

```bash
rag-ingest --dry-run                       # show resolved config, embed nothing
rag-ingest --chunk-size 1200 --chunk-overlap 200
```

Ingestion prefers `data/corpus.json` — a JSON array with **one record per
article**:

```json
[
  {"article_number": 418, "text": "...", "section": "Sale", "repealed": false}
]
```

If that file is absent it falls back to reading the PDFs under `files/` page by
page. `article_number` must be an integer; records are rejected otherwise.

## Layout

```
src/rag_service/
  config.py            # every tunable knob; `as_run_params()` feeds MLflow
  extraction/          # PDF -> text (optional OCR)
  ingest/              # corpus -> chunks -> embeddings -> FAISS; citations
  rag/                 # prompts + the engine (retriever, LLM, chain)
  api/                 # FastAPI app, schemas, dependencies
  cli.py               # rag-ingest
tests/                 # API contract, citations, ingestion
reports/               # MLflow / Locust / RAGAS artifacts
```

## Configuration

Every setting is overridable by environment variable with the `RAG_` prefix
(`RAG_CHUNK_SIZE`, `RAG_EMBEDDING_MODEL`, `RAG_INDEX_DIR`, ...). See
`src/rag_service/config.py`.

## Fine-tuning (optional, separate from serving)

```bash
pip install -e ".[finetune]"
python -m finetune.trainer --model_name <hf-model-id> --data_path <dataset.json>
```
