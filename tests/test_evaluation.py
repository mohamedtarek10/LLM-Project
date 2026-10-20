import json

import pytest
from langchain.schema import Document

from rag_service.evaluation.dataset import EvalItem, build_eval_set, load_eval_set
from rag_service.evaluation.retrieval import evaluate_retrieval
from rag_service.ingest.embeddings import HashingEmbeddings


class FakeStore:
    """Returns a fixed document list, so metrics are tested in isolation."""

    def __init__(self, results: list[list[Document]]):
        self.results = results
        self.calls = 0

    def similarity_search(self, question: str, k: int = 4):
        result = self.results[self.calls]
        self.calls += 1
        return result[:k]


def article(number: int) -> Document:
    return Document(page_content=f"body {number}", metadata={"article_number": number})


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_perfect_retrieval_scores_one():
    items = [EvalItem("q1", ["Article 1"]), EvalItem("q2", ["Article 2"])]
    store = FakeStore([[article(1)], [article(2)]])

    report = evaluate_retrieval(store, items, k=4)

    assert report.hit_rate == 1.0
    assert report.recall == 1.0
    assert report.mrr == 1.0
    assert report.misses == []


def test_total_miss_scores_zero_and_records_the_question():
    items = [EvalItem("what about delivery?", ["Article 1"])]
    store = FakeStore([[article(99)]])

    report = evaluate_retrieval(store, items, k=4)

    assert report.hit_rate == 0.0
    assert report.mrr == 0.0
    assert report.misses == ["what about delivery?"]


def test_mrr_reflects_the_rank_of_the_first_correct_source():
    items = [EvalItem("q", ["Article 3"])]
    # Correct source is third -> reciprocal rank 1/3.
    store = FakeStore([[article(1), article(2), article(3)]])

    report = evaluate_retrieval(store, items, k=4)

    assert report.hit_rate == 1.0
    assert report.mrr == pytest.approx(1 / 3)


def test_precision_accounts_for_irrelevant_chunks():
    items = [EvalItem("q", ["Article 1"])]
    store = FakeStore([[article(1), article(2), article(3), article(4)]])

    report = evaluate_retrieval(store, items, k=4)

    assert report.precision == pytest.approx(0.25)


def test_recall_is_partial_when_only_some_expected_sources_return():
    items = [EvalItem("q", ["Article 1", "Article 2"])]
    store = FakeStore([[article(1), article(9)]])

    report = evaluate_retrieval(store, items, k=4)

    assert report.recall == pytest.approx(0.5)
    assert report.hit_rate == 1.0


def test_metrics_are_named_with_k():
    items = [EvalItem("q", ["Article 1"])]
    report = evaluate_retrieval(FakeStore([[article(1)]]), items, k=3)

    assert set(report.as_metrics()) == {
        "hit_rate_at_3",
        "recall_at_3",
        "mrr_at_3",
        "precision_at_3",
    }


def test_metric_names_are_accepted_by_mlflow():
    """MLflow rejects '@' in a metric name."""
    names = evaluate_retrieval(FakeStore([[article(1)]]), [EvalItem("q", ["Article 1"])], k=4)

    assert all(c.isalnum() or c in "_-./ " for name in names.as_metrics() for c in name)


def test_empty_eval_set_is_an_error():
    with pytest.raises(ValueError, match="empty"):
        evaluate_retrieval(FakeStore([]), [], k=4)


# --------------------------------------------------------------------------- #
# Eval set construction and loading
# --------------------------------------------------------------------------- #
def test_build_eval_set_grounds_each_question_in_its_source():
    docs = [
        Document(
            page_content="The seller is obliged to deliver the thing sold to the buyer promptly.",
            metadata={"article_number": 418},
        ),
        Document(
            page_content="Every person is competent to contract unless declared otherwise here.",
            metadata={"article_number": 109},
        ),
    ]

    items = build_eval_set(docs, size=10)

    assert len(items) == 2
    assert {item.expected[0] for item in items} == {"Article 418", "Article 109"}


def test_build_eval_set_skips_documents_with_no_usable_sentence():
    docs = [Document(page_content="Short.", metadata={"article_number": 1})]

    assert build_eval_set(docs, size=10) == []


def test_build_eval_set_is_deterministic():
    docs = [
        Document(
            page_content=f"This is a sufficiently long sentence for article {n} to be sampled.",
            metadata={"article_number": n},
        )
        for n in range(1, 11)
    ]

    first = build_eval_set(docs, size=5, seed=7)
    second = build_eval_set(docs, size=5, seed=7)

    assert [i.question for i in first] == [i.question for i in second]


def test_load_eval_set_round_trips(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps([{"question": "ما هي التزامات البائع؟", "expected": ["Article 418"]}]),
        encoding="utf-8",
    )

    items = load_eval_set(path)

    assert items[0].question == "ما هي التزامات البائع؟"
    assert items[0].expected == ["Article 418"]


def test_load_eval_set_accepts_a_bare_string_expectation(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps([{"question": "q", "expected": "Article 1"}]), encoding="utf-8")

    assert load_eval_set(path)[0].expected == ["Article 1"]


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ([{"question": "", "expected": ["Article 1"]}], "empty question"),
        ([{"question": "q"}], "no expected source"),
        ({"question": "q"}, "JSON array"),
    ],
    ids=["empty-question", "no-expectation", "not-an-array"],
)
def test_load_eval_set_rejects_malformed_items(tmp_path, payload, match):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_eval_set(path)


# --------------------------------------------------------------------------- #
# The offline embedding backend
# --------------------------------------------------------------------------- #
def test_hashing_embeddings_are_unit_length():
    vector = HashingEmbeddings(dim=256).embed_query("obligations of the seller")

    assert len(vector) == 256
    assert sum(v * v for v in vector) == pytest.approx(1.0)


def test_hashing_embeddings_are_deterministic_across_instances():
    assert HashingEmbeddings().embed_query("مادة") == HashingEmbeddings().embed_query("مادة")


def test_hashing_embeddings_rank_related_text_higher():
    embedder = HashingEmbeddings()
    query = embedder.embed_query("the seller must deliver the thing sold")
    related = embedder.embed_query("delivery of the thing sold by the seller")
    unrelated = embedder.embed_query("طاقة شحن المركبات الكهربائية")

    def dot(a, b):
        # Same backend, same dim -- a length mismatch would be a real bug.
        return sum(x * y for x, y in zip(a, b, strict=True))

    assert dot(query, related) > dot(query, unrelated)


def test_hashing_embeddings_handle_empty_text():
    assert HashingEmbeddings().embed_query("") == [0.0] * 512
