import pytest

from rag_service.extraction.articles import (
    articles_to_records,
    normalize_digits,
    parse_article_number,
    segment_articles,
)


# --------------------------------------------------------------------------- #
# Article number normalisation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1", 1), ("418", 418), ("٣", 3), ("١٢٣", 123), ("۴۲", 42), (" 7 ", 7)],
    ids=["ascii", "ascii-3digit", "arabic-1", "arabic-3", "extended-arabic", "padded"],
)
def test_article_numbers_normalize_to_int(raw, expected):
    result = parse_article_number(raw)
    assert result == expected
    assert isinstance(result, int)


@pytest.mark.parametrize("raw", ["", "abc", "12a", "-", "١٢a"])
def test_non_numeric_article_numbers_are_rejected(raw):
    with pytest.raises(ValueError, match="not numeric"):
        parse_article_number(raw)


def test_normalize_digits_leaves_letters_alone():
    assert normalize_digits("مادة ٥") == "مادة 5"


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
def test_segments_arabic_and_english_headings():
    pages = [
        "مادة 1\nكل شخص أهل للتعاقد.",
        "Article 2\nThe seller must deliver.",
        "المادة ٣\nنص المادة الثالثة.",
    ]

    records = articles_to_records(segment_articles(pages))

    assert [r["article_number"] for r in records] == [1, 2, 3]
    assert all(isinstance(r["article_number"], int) for r in records)


def test_arabic_text_is_preserved_verbatim():
    body = "كل شخص أهل للتعاقد ما لم تسلب أهليته"
    records = articles_to_records(segment_articles([f"مادة 109\n{body}"]))

    assert records[0]["text"] == body


def test_repealed_articles_are_flagged():
    pages = [
        "Article 1\nThis one is in force.",
        "Article 2\nThis article was repealed in 1952.",
        "مادة 3\nهذه المادة ملغاة.",
    ]

    records = articles_to_records(segment_articles(pages))
    flags = {r["article_number"]: r["repealed"] for r in records}

    assert flags == {1: False, 2: True, 3: True}


def test_records_are_sorted_by_article_number():
    pages = ["Article 10\nTen.", "Article 2\nTwo.", "Article 7\nSeven."]

    records = articles_to_records(segment_articles(pages))

    assert [r["article_number"] for r in records] == [2, 7, 10]


def test_duplicate_headings_keep_the_longest_body():
    pages = ["Article 5\nSee below.", "Article 5\nThe full and definitive text of article five."]

    records = articles_to_records(segment_articles(pages))

    assert len(records) == 1
    assert "definitive" in records[0]["text"]


def test_bis_articles_are_kept_separate():
    pages = ["Article 5\nThe base article.", "Article 5 bis\nThe added article."]

    records = articles_to_records(segment_articles(pages))

    assert len(records) == 2
    assert {r.get("suffix") for r in records} == {None, "bis"}


def test_article_reports_the_page_it_came_from():
    pages = ["Article 1\nFirst.", "Article 2\nSecond.", "Article 3\nThird."]

    records = articles_to_records(segment_articles(pages))

    assert [r["source_pages"] for r in records] == [[1], [2], [3]]


def test_headings_with_no_body_are_skipped():
    records = articles_to_records(segment_articles(["Article 1\n", "Article 2\nReal body."]))

    assert [r["article_number"] for r in records] == [2]


def test_document_with_no_articles_yields_nothing():
    pages = ["Just a technical manual.\nNo legal articles here.", "Chapter 1: Overview"]

    assert articles_to_records(segment_articles(pages)) == []
