# LLM-Project — RAG Q&A Service

Retrieval-augmented question answering over a structured document corpus.
Answers cite their sources as **article citations**, not chunk ids.

## Run the Q&A service (3 commands)

```bash
dvc pull
docker build -t rag-service .
docker run -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY rag-service
```

`dvc pull` fetches the corpus and the prebuilt vector store, which are tracked
with DVC rather than committed to git. They are then baked into the image, so
the running container needs no data download and no index build.

`OPENAI_API_KEY` is needed for the generative answer. Retrieval itself needs no
key when the index was built with the offline `hashing` backend, which is the
default in `params.yaml` — the index records which model built it and the
service matches it automatically.

Check it is up:

```bash
curl -s localhost:8000/health
```

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
pip install -e ".[dev,ingest,ui,tracking]"
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

## Data versioning (DVC)

The source PDF and the derived article JSON are both tracked with DVC, so the
repository holds pointers and the storage remote holds the bytes.

```bash
dvc pull      # fetch the tracked PDF, the derived corpus JSON and the index
dvc repro     # rebuild the JSON from the PDF, then re-index the corpus
dvc metrics show
```

Two stages, defined in `dvc.yaml`:

| Stage | Command | In | Out |
| ----- | ------- | -- | --- |
| `extract` | `rag-extract` | `files/*.pdf` | `data/corpus.json` |
| `index`   | `rag-ingest`  | `data/corpus.json` | `faiss_ev/` |

`dvc repro` re-runs a stage only when its dependencies or its `params.yaml`
entries change, so changing `chunking.chunk_size` re-indexes without re-parsing
the PDF.

The default remote is a local directory, which is enough to demonstrate
`dvc push` / `dvc pull`. Point it at shared storage for real use:

```bash
dvc remote add -d storage s3://your-bucket/rag-service
```

### The corpus format

`rag-extract` segments the PDF into one record per article, finding both
Arabic (`مادة 12`, `المادة ١٢`) and English (`Article 12`) headings:

```json
[
  {"article_number": 418, "text": "...", "repealed": false, "source_pages": [92]}
]
```

Article numbers are normalised to integers (Arabic-Indic digits included) and a
non-numeric heading is rejected rather than entering the corpus as a string.
Articles whose body carries a repeal marker (`ملغاة`, `repealed`) are flagged,
and the flag shows up in the citation as `Article 418 (repealed)`.

## Experiment tracking (MLflow)

```bash
rag-sweep --chunk-sizes 500 1000 2000 --overlaps 100 400
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
```

Each configuration is one MLflow run, logging `chunk_size`, `overlap`,
`embedding_model`, `retriever_k` and `score_threshold` as params, and retrieval
quality as metrics:

| Metric | Meaning |
| ------ | ------- |
| `hit_rate_at_k`  | share of questions whose expected source is in the top k |
| `recall_at_k`    | share of expected sources retrieved, per question |
| `mrr_at_k`       | 1 / rank of the first correct source |
| `precision_at_k` | share of retrieved chunks that were expected |

Runs are ranked on `hit_rate`, with `mrr` breaking ties, and the winner is
registered in the MLflow Model Registry as **`rag-chunking-best`**. The
registered artifact carries the FAISS index that config produced, so the
registered version retrieves exactly as the sweep measured.

Comparison screenshots live in [`reports/`](reports/).

### Why the sweep needs no API key

The sweep scores *retrieval*, not generation, so it never calls an LLM. The
`embedding_model` setting selects the backend:

| Value | Backend |
| ----- | ------- |
| `hashing` | deterministic hashed bag-of-words — offline, no key, used by CI |
| `hf:<model-id>` | local sentence-transformers |
| `text-embedding-3-small`, ... | OpenAI |

`hashing` is a lexical baseline, not a semantic model. Use it to compare
chunking mechanics reproducibly; re-run the sweep with an OpenAI or HF model
for numbers that reflect production quality:

```bash
rag-sweep --embedding-models text-embedding-3-small
```

### The evaluation set

`data/eval_questions.json` pairs each question with the citation that should
come back. If it is absent, the sweep derives one from the corpus by taking a
distinctive sentence out of each record and asking which record it came from —
a real retrieval test that needs no LLM, so it can gate CI.

## CI

`.github/workflows/ci.yml` runs five jobs in sequence; each one gates the next.

| Job | Does | Needs a secret |
| --- | ---- | -------------- |
| `lint` | `ruff check` and `ruff format --check` | no |
| `test` | the pytest suite, uploading a JUnit report | no |
| `index` | rebuilds the index and checks it answers a query | optional |
| `faithfulness` | **fails the build if RAGAS faithfulness < 0.75** | `OPENAI_API_KEY` |
| `image` | builds and pushes to GHCR, then smoke-tests it | no (uses `GITHUB_TOKEN`) |

Lint, test and the index rebuild need no key: `RAG_EMBEDDING_MODEL=hashing`
puts CI on the deterministic offline backend.

### The faithfulness gate

`rag-eval` asks the live pipeline the 20 questions in
`data/ragas_questions.json`, then scores each answer with RAGAS faithfulness —
is every claim in the answer supported by the passages actually retrieved? The
command exits non-zero when the mean falls below the threshold, which is what
fails the job:

```bash
rag-eval --questions data/ragas_questions.json --min-faithfulness 0.75
```

Two deliberate choices:

- A question that retrieved nothing gets **no score** rather than a zero.
  Scoring an honest "not in the documents" reply as unfaithful would push the
  pipeline toward confident guessing, which is the failure the gate exists to
  catch.
- If **nothing** could be scored the gate **fails**. An empty result must never
  pass by vacuous truth.

RAGAS judges answers with an LLM, so this job needs `OPENAI_API_KEY` under
*Settings → Secrets and variables → Actions*. Without it the job fails with an
explicit message rather than passing silently. It is skipped for pull requests
from forks, which cannot read secrets.

### Data in CI

The real corpus lives behind the DVC remote, which a runner cannot reach by
default. Set `DVC_REMOTE_URL` (plus cloud credentials) as secrets and the
`index` job runs `dvc pull && dvc repro` for real. Without them it falls back
to `tests/fixtures/corpus.json`, a 12-article fixture that still exercises the
article-segmented path end to end, and says so in the job log.

## Layout

```
src/rag_service/
  config.py            # every tunable knob; `as_run_params()` feeds MLflow
  extraction/          # PDF -> text (optional OCR)
  ingest/              # corpus -> chunks -> embeddings -> FAISS; citations
  rag/                 # prompts + the engine (retriever, LLM, chain)
  api/                 # FastAPI app, schemas, dependencies
  evaluation/          # retrieval metrics + the RAGAS faithfulness gate
  tracking/            # the MLflow sweep and registered model
  cli.py               # rag-extract / rag-ingest / rag-sweep / rag-eval
tests/                 # API contract, citations, ingestion, gate logic
  fixtures/corpus.json # 12-article corpus CI builds from
reports/               # MLflow / Locust / RAGAS artifacts
.github/workflows/     # lint -> test -> index -> faithfulness -> image
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
