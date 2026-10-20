from langchain.schema import Document

from rag_service.ingest.citations import format_citation, unique_citations


def test_article_citation_prefers_article_number():
    assert format_citation({"article_number": 418}) == "Article 418"


def test_article_citation_flags_repealed_articles():
    citation = format_citation({"article_number": 7, "repealed": True})
    assert citation == "Article 7 (repealed)"


def test_article_citation_includes_section():
    citation = format_citation({"article_number": 12, "section": "Obligations"})
    assert citation == "Article 12 — Obligations"


def test_falls_back_to_file_and_page():
    assert format_citation({"file_name": "code.pdf", "page": 3}) == "code.pdf, p. 3"


def test_handles_missing_metadata():
    assert format_citation(None) == "unknown source"
    assert format_citation({}) == "unknown source"


def test_citations_never_expose_chunk_ids():
    meta = {"article_number": 5, "chunk_id": "abc-123", "_id": "xyz"}
    assert "abc-123" not in format_citation(meta)
    assert "xyz" not in format_citation(meta)


def test_unique_citations_dedupes_and_preserves_order():
    docs = [
        Document(page_content="a", metadata={"article_number": 2}),
        Document(page_content="b", metadata={"article_number": 1}),
        Document(page_content="c", metadata={"article_number": 2}),
    ]
    assert unique_citations(docs) == ["Article 2", "Article 1"]


def test_unique_citations_handles_no_documents():
    assert unique_citations(None) == []
