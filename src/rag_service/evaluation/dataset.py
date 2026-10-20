"""The retrieval evaluation set.

Comparing chunking configs needs ground truth: for a given question, which
source *should* come back. Hand-written questions are better, but a corpus can
also supply them -- take a distinctive sentence out of a record and ask which
record it came from. That is a real retrieval test (it measures whether
chunking kept the passage findable) and it needs no LLM and no API key, so it
can gate CI.
"""

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from rag_service.ingest.citations import format_citation

_SENTENCE = re.compile(r"(?<=[.!؟?])\s+|\n+")
MIN_SENTENCE_CHARS = 40


@dataclass
class EvalItem:
    question: str
    expected: list[str]  # acceptable citations

    def to_record(self) -> dict:
        return {"question": self.question, "expected": self.expected}


def load_eval_set(path: Path | str) -> list[EvalItem]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a JSON array of eval items")

    items = []
    for index, record in enumerate(records):
        question = (record.get("question") or "").strip()
        if not question:
            raise ValueError(f"{path} item {index}: empty question")
        expected = record.get("expected") or record.get("expected_sources") or []
        if isinstance(expected, str):
            expected = [expected]
        if not expected:
            raise ValueError(f"{path} item {index}: no expected source")
        items.append(EvalItem(question=question, expected=list(expected)))
    return items


def _pick_sentence(text: str) -> str | None:
    """The longest sentence in a record -- the most distinctive query."""
    candidates = [s.strip() for s in _SENTENCE.split(text) if len(s.strip()) >= MIN_SENTENCE_CHARS]
    return max(candidates, key=len) if candidates else None


def build_eval_set(documents, size: int = 20, seed: int = 13) -> list[EvalItem]:
    """Build an eval set by sampling a distinctive sentence per document."""
    rng = random.Random(seed)
    pool = []
    for doc in documents:
        sentence = _pick_sentence(doc.page_content)
        if sentence:
            pool.append(EvalItem(question=sentence, expected=[format_citation(doc.metadata)]))

    rng.shuffle(pool)
    return pool[:size]


def save_eval_set(items: list[EvalItem], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([item.to_record() for item in items], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
