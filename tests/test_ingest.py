import json

import pytest

from rag_service.config import Settings
from rag_service.ingest.chunking import split_articles
from rag_service.ingest.pipeline import load_corpus_records


def write_corpus(tmp_path, records):
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return path


def test_loads_one_document_per_article(tmp_path):
    path = write_corpus(
        tmp_path,
        [
            {"article_number": 1, "text": "First article.", "section": "General"},
            {"article_number": 2, "text": "Second article.", "repealed": True},
        ],
    )

    docs = load_corpus_records(path)

    assert len(docs) == 2
    assert docs[0].page_content == "First article."
    assert docs[0].metadata["article_number"] == 1
    assert docs[0].metadata["repealed"] is False
    assert docs[1].metadata["repealed"] is True


def test_preserves_arabic_text(tmp_path):
    arabic = "كل شخص أهل للتعاقد ما لم تسلب أهليته"
    path = write_corpus(tmp_path, [{"article_number": 109, "text": arabic}])

    assert load_corpus_records(path)[0].page_content == arabic


def test_skips_records_with_no_text(tmp_path):
    path = write_corpus(
        tmp_path,
        [{"article_number": 1, "text": "   "}, {"article_number": 2, "text": "Real."}],
    )
    assert len(load_corpus_records(path)) == 1


def test_rejects_non_integer_article_numbers(tmp_path):
    path = write_corpus(tmp_path, [{"article_number": "418", "text": "Body."}])

    with pytest.raises(ValueError, match="must be an int"):
        load_corpus_records(path)


def test_rejects_corpus_that_is_not_an_array(tmp_path):
    path = tmp_path / "corpus.json"
    path.write_text('{"article_number": 1, "text": "x"}', encoding="utf-8")

    with pytest.raises(ValueError, match="JSON array"):
        load_corpus_records(path)


def test_short_articles_are_not_split(tmp_path):
    """One article should stay one chunk, so its citation is not scattered."""
    path = write_corpus(
        tmp_path, [{"article_number": n, "text": f"Short article {n}."} for n in range(1, 6)]
    )
    docs = load_corpus_records(path)

    chunks = split_articles(docs, Settings(chunk_size=500, chunk_overlap=50))

    assert len(chunks) == 5


def test_overlong_articles_are_split_but_keep_article_metadata(tmp_path):
    long_text = ". ".join(f"sentence number {i}" for i in range(200))
    path = write_corpus(tmp_path, [{"article_number": 99, "text": long_text}])
    docs = load_corpus_records(path)

    chunks = split_articles(docs, Settings(chunk_size=200, chunk_overlap=20))

    assert len(chunks) > 1
    assert all(chunk.metadata["article_number"] == 99 for chunk in chunks)
