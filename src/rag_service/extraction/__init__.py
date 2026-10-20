"""Document extraction: PDF -> page text -> article records.

``PyMuPDFImageReader`` is imported lazily so that the pure-Python article
segmentation stays usable without the ``ingest`` extra installed.
"""

from rag_service.extraction.articles import (
    Article,
    articles_to_records,
    normalize_digits,
    parse_article_number,
    segment_articles,
)

__all__ = [
    "Article",
    "PyMuPDFImageReader",
    "articles_to_records",
    "normalize_digits",
    "parse_article_number",
    "segment_articles",
]


def __getattr__(name: str):
    if name == "PyMuPDFImageReader":
        from rag_service.extraction.pdf_reader import PyMuPDFImageReader

        return PyMuPDFImageReader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
