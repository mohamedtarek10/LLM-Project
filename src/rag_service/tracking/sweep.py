"""The chunking sweep: one MLflow run per configuration.

Each run logs the config as params (chunk_size, overlap, embedding_model, ...)
and retrieval quality as metrics, so runs are directly comparable in the MLflow
UI. The best run's config is registered in the Model Registry.
"""

import json
import tempfile
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import mlflow

from rag_service.config import Settings, get_settings
from rag_service.evaluation.dataset import build_eval_set, load_eval_set, save_eval_set
from rag_service.evaluation.retrieval import RetrievalReport, evaluate_retrieval
from rag_service.ingest.chunking import split_articles, split_documents
from rag_service.ingest.embeddings import get_embeddings
from rag_service.ingest.pipeline import load_corpus_records, load_pdf_documents
from rag_service.tracking.model import RetrieverModel

EXPERIMENT_NAME = "rag-chunking"
REGISTERED_MODEL_NAME = "rag-chunking-best"


@dataclass
class RunResult:
    chunk_size: int
    overlap: int
    embedding_model: str
    report: RetrievalReport
    chunks: int
    run_id: str

    @property
    def score(self) -> tuple[float, float]:
        """Rank on hit_rate; mrr breaks ties.

        A correct source at rank 1 beats the same source at rank 4 for an
        answer that has to cite it.
        """
        return (self.report.hit_rate, self.report.mrr)


def _load_documents(settings: Settings):
    """The corpus, article-segmented when the derived JSON exists."""
    if settings.corpus_json.exists():
        records = load_corpus_records(settings.corpus_json)
        if records:
            return records, True
        print(f"{settings.corpus_json} holds no articles; using page-level documents")

    documents = load_pdf_documents(settings.source_dir)
    if not documents:
        raise RuntimeError(
            f"No corpus at {settings.corpus_json} and no PDFs under {settings.source_dir}."
        )
    return documents, False


def _eval_items(settings: Settings, documents, size: int):
    """Use the committed eval set when present, else derive one and save it."""
    if settings.eval_set.exists():
        return load_eval_set(settings.eval_set), False

    items = build_eval_set(documents, size=size)
    save_eval_set(items, settings.eval_set)
    return items, True


def run_sweep(
    chunk_sizes: list[int],
    overlaps: list[int],
    embedding_models: list[str] | None = None,
    settings: Settings | None = None,
    eval_size: int = 20,
    tracking_uri: str | None = None,
    register: bool = True,
) -> list[RunResult]:
    """One MLflow run per (chunk_size, overlap, embedding_model) combination."""
    base = settings or get_settings()
    embedding_models = embedding_models or [base.embedding_model]

    mlflow.set_tracking_uri(tracking_uri or base.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    documents, article_segmented = _load_documents(base)
    items, generated = _eval_items(base, documents, eval_size)
    print(
        f"Corpus: {len(documents)} documents "
        f"({'article-segmented' if article_segmented else 'page-level'}); "
        f"eval set: {len(items)} questions{' (generated)' if generated else ''}"
    )

    splitter = split_articles if article_segmented else split_documents
    results: list[RunResult] = []

    for chunk_size, overlap, embedding_model in product(chunk_sizes, overlaps, embedding_models):
        if overlap >= chunk_size:
            print(f"Skipping overlap {overlap} >= chunk_size {chunk_size}")
            continue

        config = base.model_copy(
            update={
                "chunk_size": chunk_size,
                "chunk_overlap": overlap,
                "embedding_model": embedding_model,
            }
        )
        chunks = splitter(documents, config)
        run_name = f"cs{chunk_size}-ov{overlap}-{embedding_model.replace(':', '_')}"

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(config.as_run_params())
            mlflow.log_params(
                {
                    "documents": len(documents),
                    "chunks": len(chunks),
                    "article_segmented": article_segmented,
                    "eval_questions": len(items),
                }
            )

            from langchain_community.vectorstores import FAISS

            vectorstore = FAISS.from_documents(chunks, embedding=get_embeddings(config))
            report = evaluate_retrieval(vectorstore, items, k=config.retriever_k)

            mlflow.log_metrics(report.as_metrics())
            mlflow.log_metrics(
                {
                    "chunks": float(len(chunks)),
                    "avg_chunk_chars": sum(len(c.page_content) for c in chunks) / len(chunks),
                }
            )

            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                index_dir = tmp_path / "index"
                vectorstore.save_local(str(index_dir))

                config_path = tmp_path / "config.json"
                config_path.write_text(
                    json.dumps(config.as_run_params(), indent=2), encoding="utf-8"
                )

                misses_path = tmp_path / "misses.json"
                misses_path.write_text(
                    json.dumps(report.misses, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                mlflow.log_artifact(str(misses_path), artifact_path="evaluation")

                mlflow.pyfunc.log_model(
                    name="retriever",
                    python_model=RetrieverModel(),
                    artifacts={"index": str(index_dir), "config": str(config_path)},
                )

            results.append(
                RunResult(
                    chunk_size=chunk_size,
                    overlap=overlap,
                    embedding_model=embedding_model,
                    report=report,
                    chunks=len(chunks),
                    run_id=run.info.run_id,
                )
            )
            print(
                f"  {run_name}: chunks={len(chunks)} "
                f"hit_rate={report.hit_rate:.3f} mrr={report.mrr:.3f}"
            )

    if not results:
        raise RuntimeError("The sweep produced no runs; check the grid.")

    best = max(results, key=lambda r: r.score)
    print(
        f"\nBest: chunk_size={best.chunk_size} overlap={best.overlap} "
        f"embedding_model={best.embedding_model} "
        f"(hit_rate={best.report.hit_rate:.3f}, mrr={best.report.mrr:.3f})"
    )

    if register:
        version = mlflow.register_model(
            model_uri=f"runs:/{best.run_id}/retriever",
            name=REGISTERED_MODEL_NAME,
        )
        print(f"Registered {REGISTERED_MODEL_NAME} version {version.version}")

    return results
