"""Chunking strategy. The parameters here are what MLflow runs compare."""

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from rag_service.config import Settings, get_settings


def make_splitter(settings: Settings | None = None) -> RecursiveCharacterTextSplitter:
    settings = settings or get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def split_documents(documents: list[Document], settings: Settings | None = None) -> list[Document]:
    """Split free-form documents (PDF pages, web pages) into chunks."""
    return make_splitter(settings).split_documents(documents)


def split_articles(documents: list[Document], settings: Settings | None = None) -> list[Document]:
    """Chunk an article-segmented corpus.

    One article is the natural unit, so an article short enough to fit is kept
    whole -- splitting it would scatter a single citation across chunks. Only
    over-long articles are split, and each piece keeps the article metadata.
    """
    settings = settings or get_settings()
    splitter = make_splitter(settings)

    chunks: list[Document] = []
    for doc in documents:
        if len(doc.page_content) <= settings.chunk_size:
            chunks.append(doc)
        else:
            chunks.extend(splitter.split_documents([doc]))
    return chunks
