"""Retrieval metrics -- the numbers that rank one chunking config against another.

Retrieval only: no generation, so a sweep costs nothing beyond embeddings and
is deterministic with the offline backend.
"""

from dataclasses import dataclass, field

from rag_service.evaluation.dataset import EvalItem
from rag_service.ingest.citations import format_citation


@dataclass
class RetrievalReport:
    k: int
    hit_rate: float
    recall: float
    mrr: float
    precision: float
    questions: int
    misses: list[str] = field(default_factory=list)

    def as_metrics(self) -> dict[str, float]:
        """Metric names for MLflow, which rejects '@' in a metric name."""
        return {
            f"hit_rate_at_{self.k}": self.hit_rate,
            f"recall_at_{self.k}": self.recall,
            f"mrr_at_{self.k}": self.mrr,
            f"precision_at_{self.k}": self.precision,
        }


def evaluate_retrieval(vectorstore, items: list[EvalItem], k: int = 4) -> RetrievalReport:
    """Score the vector store against the eval set.

    * hit_rate  -- share of questions with at least one expected source in top-k
    * recall    -- share of expected sources retrieved, averaged per question
    * mrr       -- 1/rank of the first correct source, averaged
    * precision -- share of retrieved chunks that were expected, averaged
    """
    if not items:
        raise ValueError("The eval set is empty; nothing to measure.")

    hits = reciprocal_ranks = recalls = precisions = 0.0
    misses: list[str] = []

    for item in items:
        retrieved = vectorstore.similarity_search(item.question, k=k)
        citations = [format_citation(doc.metadata) for doc in retrieved]
        expected = set(item.expected)

        matched = expected.intersection(citations)
        if matched:
            hits += 1
            first = next(i for i, c in enumerate(citations, start=1) if c in expected)
            reciprocal_ranks += 1 / first
        else:
            misses.append(item.question[:80])

        recalls += len(matched) / len(expected)
        precisions += (
            sum(1 for c in citations if c in expected) / len(citations) if citations else 0.0
        )

    total = len(items)
    return RetrievalReport(
        k=k,
        hit_rate=hits / total,
        recall=recalls / total,
        mrr=reciprocal_ranks / total,
        precision=precisions / total,
        questions=total,
        misses=misses,
    )
