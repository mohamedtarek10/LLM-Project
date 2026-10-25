"""Command line entry points.

Three commands, each one a DVC stage or an experiment driver:

* ``rag-extract`` -- source PDF -> ``data/corpus.json`` (one record per article)
* ``rag-ingest``  -- corpus JSON -> FAISS index
* ``rag-sweep``   -- the MLflow chunking sweep

Parameters resolve in this order: command-line flag, then ``params.yaml`` (so
``dvc repro`` reacts to a params change), then the ``RAG_*`` environment, then
the defaults in ``config.py``.
"""

import argparse
import json
from pathlib import Path

from rag_service.config import PROJECT_ROOT, Settings, get_settings

PARAMS_FILE = PROJECT_ROOT / "params.yaml"


def load_params() -> dict:
    """Read ``params.yaml`` if it exists; DVC owns this file."""
    if not PARAMS_FILE.exists():
        return {}
    try:
        import yaml
    except ImportError:
        print("params.yaml present but PyYAML is not installed; ignoring it.")
        return {}
    return yaml.safe_load(PARAMS_FILE.read_text(encoding="utf-8")) or {}


def settings_from_params(**overrides) -> Settings:
    """Settings with params.yaml applied, then explicit overrides."""
    params = load_params()
    update: dict[str, object] = {}

    chunking = params.get("chunking") or {}
    if "chunk_size" in chunking:
        update["chunk_size"] = chunking["chunk_size"]
    if "chunk_overlap" in chunking:
        update["chunk_overlap"] = chunking["chunk_overlap"]

    embedding = params.get("embedding") or {}
    if "model" in embedding:
        update["embedding_model"] = embedding["model"]

    retrieval = params.get("retrieval") or {}
    if "k" in retrieval:
        update["retriever_k"] = retrieval["k"]

    update.update({k: v for k, v in overrides.items() if v is not None})
    return get_settings().model_copy(update=update)


# --------------------------------------------------------------------------- #
# rag-extract
# --------------------------------------------------------------------------- #
def extract() -> None:
    """Segment the source PDF(s) into one JSON record per article."""
    settings = settings_from_params()
    parser = argparse.ArgumentParser(description="Extract articles from the source PDF(s)")
    parser.add_argument("--source-dir", type=Path, default=settings.source_dir)
    parser.add_argument("--output", type=Path, default=settings.corpus_json)
    parser.add_argument("--ocr", action="store_true", help="OCR image-only pages")
    parser.add_argument(
        "--min-articles",
        type=int,
        default=10,
        help="Below this count the document is treated as not article-structured.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help=(
            "Write an empty corpus instead of failing when the document is not "
            "article-structured; the pipeline then indexes it page by page."
        ),
    )
    args = parser.parse_args()

    from rag_service.extraction import PyMuPDFImageReader, articles_to_records, segment_articles

    pdfs = sorted(Path(args.source_dir).rglob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found under {args.source_dir}")

    reader = PyMuPDFImageReader(extract_images=args.ocr)
    records: list[dict] = []
    total_pages = 0
    for pdf in pdfs:
        pages = reader.read_pages(pdf)
        total_pages += len(pages)
        articles = segment_articles(pages)
        print(f"{pdf.name}: {len(pages)} pages -> {len(articles)} articles")
        for record in articles_to_records(articles):
            record["file_name"] = pdf.name
            records.append(record)

    # A handful of matches across a long document means the regex caught prose
    # like "see Article 5", not real headings. Writing those would poison the
    # corpus with fragments, so treat the document as not article-structured.
    if len(records) < args.min_articles:
        message = (
            f"Found only {len(records)} article heading(s) across {total_pages} pages, "
            f"below --min-articles={args.min_articles}. This document does not look "
            f"article-structured."
        )
        if not args.allow_empty:
            raise SystemExit(
                message
                + " Re-run with --allow-empty to index it page by page"
                + " instead, or lower --min-articles if the count is right."
            )
        print(f"{message} Writing an empty corpus; the pipeline will index pages.")
        records = []

    repealed = sum(1 for r in records if r.get("repealed"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} articles ({repealed} flagged repealed) to {args.output}")


# --------------------------------------------------------------------------- #
# rag-ingest
# --------------------------------------------------------------------------- #
def main() -> None:
    """Build the vector index from the corpus."""
    defaults = settings_from_params()
    parser = argparse.ArgumentParser(description="Build the RAG vector index")
    parser.add_argument("--chunk-size", type=int, default=defaults.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=defaults.chunk_overlap)
    parser.add_argument("--embedding-model", default=defaults.embedding_model)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the resolved configuration and exit without embedding.",
    )
    args = parser.parse_args()

    settings = defaults.model_copy(
        update={
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "embedding_model": args.embedding_model,
        }
    )

    if args.dry_run:
        print("Resolved configuration:")
        for key, value in settings.as_run_params().items():
            print(f"  {key}: {value}")
        print(f"  corpus_json: {settings.corpus_json} (exists={settings.corpus_json.exists()})")
        print(f"  source_dir:  {settings.source_dir} (exists={settings.source_dir.exists()})")
        print(f"  index_dir:   {settings.index_dir}")
        return

    from rag_service.ingest.pipeline import build_index, count_documents

    vectorstore = build_index(settings)
    print(f"documents_indexed={count_documents(vectorstore)}")


# --------------------------------------------------------------------------- #
# rag-sweep
# --------------------------------------------------------------------------- #
def sweep() -> None:
    """Run the MLflow chunking sweep."""
    defaults = settings_from_params()
    parser = argparse.ArgumentParser(description="Sweep chunking configs, logging to MLflow")
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[500, 1000, 2000])
    parser.add_argument("--overlaps", type=int, nargs="+", default=[100, 400])
    parser.add_argument(
        "--embedding-models",
        nargs="+",
        default=[defaults.embedding_model],
        help="Model names: 'hashing', 'hf:<id>', or an OpenAI embedding model.",
    )
    parser.add_argument("--eval-size", type=int, default=20)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    from rag_service.tracking import run_sweep

    run_sweep(
        chunk_sizes=args.chunk_sizes,
        overlaps=args.overlaps,
        embedding_models=args.embedding_models,
        settings=defaults,
        eval_size=args.eval_size,
        tracking_uri=args.tracking_uri,
        register=not args.no_register,
    )


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------- #
# rag-eval
# --------------------------------------------------------------------------- #
def evaluate_faithfulness_cli() -> None:
    """Run the RAGAS faithfulness gate. Exits non-zero below the threshold."""
    import sys

    defaults = settings_from_params()
    parser = argparse.ArgumentParser(
        description="Score answer faithfulness with RAGAS and gate on it"
    )
    parser.add_argument(
        "--questions", type=Path, default=PROJECT_ROOT / "data" / "ragas_questions.json"
    )
    parser.add_argument(
        "--min-faithfulness",
        type=float,
        default=None,
        help="Fail if mean faithfulness falls below this (default: 0.75).",
    )
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "reports" / "ragas.json")
    args = parser.parse_args()

    from rag_service.evaluation.ragas_eval import (
        DEFAULT_MIN_FAITHFULNESS,
        evaluate_faithfulness,
        load_questions,
    )
    from rag_service.rag import RagEngine

    threshold = (
        args.min_faithfulness if args.min_faithfulness is not None else DEFAULT_MIN_FAITHFULNESS
    )
    questions = load_questions(args.questions)
    report = evaluate_faithfulness(RagEngine(defaults), questions, threshold=threshold)

    path = report.write(args.report)
    print(f"\n{report.summary()}")
    print(f"Report written to {path}")

    if not report.passed:
        sys.exit(1)
