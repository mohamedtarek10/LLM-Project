"""Command line entry point for (re)building the index."""

import argparse

from rag_service.config import get_settings
from rag_service.ingest.pipeline import build_index, count_documents


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build the RAG vector index")
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    parser.add_argument("--embedding-model", default=settings.embedding_model)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the resolved configuration and exit without embedding.",
    )
    args = parser.parse_args()

    settings.chunk_size = args.chunk_size
    settings.chunk_overlap = args.chunk_overlap
    settings.embedding_model = args.embedding_model

    if args.dry_run:
        print("Resolved configuration:")
        for key, value in settings.as_run_params().items():
            print(f"  {key}: {value}")
        print(f"  corpus_json: {settings.corpus_json} (exists={settings.corpus_json.exists()})")
        print(f"  source_dir:  {settings.source_dir} (exists={settings.source_dir.exists()})")
        print(f"  index_dir:   {settings.index_dir}")
        return

    vectorstore = build_index(settings)
    print(f"documents_indexed={count_documents(vectorstore)}")


if __name__ == "__main__":
    main()
