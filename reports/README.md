# Reports

Artifacts from the chunking sweep. Regenerate with:

```bash
rag-sweep --chunk-sizes 500 1000 2000 --overlaps 100 400
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Screenshots

| File | Shows |
| ---- | ----- |
| `mlflow-runs-metrics.jpg` | the 6 runs with `hit_rate_at_4` and `mrr_at_4` |
| `mlflow-runs-params.jpg` | the same runs with the `chunk_size` parameter |
| `mlflow-model-registry.jpg` | `rag-chunking-best` version 1 in the Model Registry |

## Run comparison

Six configurations, 20 generated eval questions, `k=4`, `embedding_model=hashing`.
Ranked on `hit_rate`, with `mrr` breaking ties.

| chunk_size | overlap | chunks | hit_rate@4 | mrr@4 |
| ---------: | ------: | -----: | ---------: | ----: |
| **500**    | **100** |    349 |  **0.650** | **0.617** |
| 500        | 400     |   1080 |      0.650 | 0.617 |
| 1000       | 100     |    175 |      0.650 | 0.583 |
| 1000       | 400     |    209 |      0.650 | 0.583 |
| 2000       | 100     |    107 |      0.600 | 0.567 |
| 2000       | 400     |    108 |      0.600 | 0.533 |

**Registered:** `chunk_size=500, overlap=100` as `rag-chunking-best` v1. It ties
with `overlap=400` on both metrics but produces 349 chunks instead of 1080, so
it indexes and retrieves at a third of the cost for the same quality.

## How to read these numbers

Two caveats, both worth fixing before treating this as a quality verdict:

1. **The corpus is not article-structured.** `files/UDG-Full-Doc-English.pdf` is
   an EV charging manual, so `rag-extract` finds no real article headings and
   the pipeline falls back to indexing pages. Ground truth is therefore a page
   citation, and every chunk cut from one page shares it — which is why
   `hit_rate` barely moves across configs while `mrr` does. Against an
   article-segmented corpus the spread will be wider.

2. **`hashing` is a lexical backend, not a semantic one.** It exists so the
   sweep and CI run without an API key. Re-run with a real embedding model for
   numbers that reflect production behaviour:

   ```bash
   rag-sweep --embedding-models text-embedding-3-small
   ```

Both are properties of the current inputs, not of the pipeline: the same sweep
command produces meaningful numbers once a real corpus and a real embedding
model are in place.
