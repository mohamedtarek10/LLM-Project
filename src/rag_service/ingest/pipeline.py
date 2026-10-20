"""Ingestion: source documents -> chunks -> embeddings -> FAISS.

Two entry points into the same vector store:

* ``load_corpus_records`` reads the structured JSON corpus (one record per
  article). This is the path the project is meant to use.
* ``load_pdf_documents`` reads raw PDFs page by page, for corpora that have not
  been article-segmented yet.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from rag_service.config import Settings, get_settings
from rag_service.ingest.chunking import split_articles, split_documents


def get_embeddings(settings: Settings | None = None):
    settings = settings or get_settings()
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=settings.embedding_model, openai_api_key=settings.openai_api_key)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_corpus_records(path: Path | str) -> list[Document]:
    """Load the structured corpus: a JSON array of one record per article.

    Expected record shape (extra keys are carried through as metadata)::

        {"article_number": 1, "text": "...", "section": "...", "repealed": false}
    """
    path = Path(path)
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(
            f"{path} must contain a JSON array of article records, got {type(records).__name__}"
        )

    documents = []
    for index, record in enumerate(records):
        text = record.get("text") or record.get("body") or ""
        if not text.strip():
            continue

        article = record.get("article_number")
        if article is not None and not isinstance(article, int):
            raise ValueError(
                f"{path} record {index}: article_number must be an int, got {article!r}"
            )

        metadata = {k: v for k, v in record.items() if k not in {"text", "body"}}
        metadata.setdefault("file_name", path.name)
        metadata.setdefault("repealed", False)
        documents.append(Document(page_content=text, metadata=metadata))

    return documents


def load_pdf_documents(target_directory: Path | str) -> list[Document]:
    """Read every PDF under ``target_directory`` into page-level documents."""
    from rag_service.extraction import PyMuPDFImageReader

    paths = [
        os.path.join(root, name)
        for root, _dirs, files in os.walk(target_directory)
        for name in files
        if name.lower().endswith(".pdf")
    ]
    if not paths:
        return []

    print(f"Found {len(paths)} PDF(s): {[os.path.basename(p) for p in paths]}")
    reader = PyMuPDFImageReader(extract_images=False)

    documents: list[Document] = []
    for path in paths:
        for doc in reader.load_data(path):
            documents.append(doc.to_langchain_format())
    return documents


# --------------------------------------------------------------------------- #
# Index building
# --------------------------------------------------------------------------- #
def build_index(settings: Settings | None = None, save: bool = True) -> FAISS:
    """Build the vector store from the best available source.

    Prefers the structured corpus JSON; falls back to the raw PDFs.
    """
    settings = settings or get_settings()

    if settings.corpus_json.exists():
        print(f"Loading structured corpus from {settings.corpus_json}")
        documents = load_corpus_records(settings.corpus_json)
        chunks = split_articles(documents, settings)
    else:
        print(
            f"No corpus JSON at {settings.corpus_json}; falling back to PDFs in "
            f"{settings.source_dir}"
        )
        documents = load_pdf_documents(settings.source_dir)
        chunks = split_documents(documents, settings)

    if not chunks:
        raise RuntimeError(
            "Nothing to index. Provide either a corpus JSON or PDFs under the source dir."
        )

    print(f"Embedding {len(chunks)} chunks from {len(documents)} documents...")
    vectorstore = FAISS.from_documents(documents=chunks, embedding=get_embeddings(settings))

    if save:
        settings.index_dir.parent.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(settings.index_dir))
        print(f"Saved index to {settings.index_dir}")
    return vectorstore


def load_index(settings: Settings | None = None) -> FAISS:
    """Load the persisted vector store, building it if it is missing."""
    settings = settings or get_settings()
    if not settings.index_dir.exists():
        return build_index(settings)

    print(f"Loading FAISS index from {settings.index_dir}")
    return FAISS.load_local(
        str(settings.index_dir),
        embeddings=get_embeddings(settings),
        allow_dangerous_deserialization=True,
    )


def add_url(url: str, vectorstore: FAISS, settings: Settings | None = None) -> int:
    """Scrape a URL into the existing store. Returns the number of chunks added."""
    settings = settings or get_settings()
    from langchain_community.document_loaders import WebBaseLoader

    docs = WebBaseLoader(url).load()
    for doc in docs:
        doc.metadata["file_name"] = url
        doc.metadata["last_modified_date"] = datetime.now(timezone.utc).isoformat()

    chunks = split_documents(docs, settings)
    vectorstore.add_documents(chunks)
    vectorstore.save_local(str(settings.index_dir))
    return len(chunks)


def count_documents(vectorstore: FAISS) -> int:
    """Number of indexed vectors -- what /health reports."""
    index = getattr(vectorstore, "index", None)
    if index is not None and hasattr(index, "ntotal"):
        return int(index.ntotal)
    return len(getattr(vectorstore, "docstore", {}).__dict__.get("_dict", {}))
