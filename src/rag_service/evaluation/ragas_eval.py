"""RAGAS faithfulness evaluation and the CI gate.

Faithfulness asks: is every claim in the answer supported by the passages that
were retrieved? It is the metric worth blocking a release on, because a RAG
service that invents unsupported statements is worse than one that says it does
not know.

Unlike the retrieval metrics in ``retrieval.py``, this needs an LLM to judge the
answers, so it needs an API key and cannot run offline.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import fmean

DEFAULT_MIN_FAITHFULNESS = 0.75


# --------------------------------------------------------------------------- #
# Questions
# --------------------------------------------------------------------------- #
def load_questions(path: Path | str) -> list[str]:
    """Load the question set: a JSON array of strings or {"question": ...}."""
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a JSON array of questions")

    questions = []
    for index, record in enumerate(records):
        question = record if isinstance(record, str) else (record or {}).get("question", "")
        question = (question or "").strip()
        if not question:
            raise ValueError(f"{path} item {index}: empty question")
        questions.append(question)

    if not questions:
        raise ValueError(f"{path} contains no questions")
    return questions


# --------------------------------------------------------------------------- #
# Report and gate
# --------------------------------------------------------------------------- #
@dataclass
class QuestionScore:
    question: str
    faithfulness: float | None
    sources: list[str] = field(default_factory=list)


@dataclass
class FaithfulnessReport:
    threshold: float
    scores: list[QuestionScore]

    @property
    def scored(self) -> list[float]:
        """Scores RAGAS could actually compute.

        A question whose answer cites nothing yields no score; averaging those
        in as zero would punish an honest "not in the documents" reply.
        """
        return [s.faithfulness for s in self.scores if s.faithfulness is not None]

    @property
    def mean(self) -> float:
        return fmean(self.scored) if self.scored else 0.0

    @property
    def unscored(self) -> list[str]:
        return [s.question for s in self.scores if s.faithfulness is None]

    @property
    def passed(self) -> bool:
        return bool(self.scored) and self.mean >= self.threshold

    @property
    def worst(self) -> list[QuestionScore]:
        scored = [s for s in self.scores if s.faithfulness is not None]
        return sorted(scored, key=lambda s: s.faithfulness)[:5]

    def as_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "mean_faithfulness": self.mean,
            "questions": len(self.scores),
            "scored": len(self.scored),
            "unscored": self.unscored,
            "passed": self.passed,
            "scores": [asdict(s) for s in self.scores],
        }

    def write(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        lines = [
            f"{verdict}  faithfulness {self.mean:.3f} "
            f"(threshold {self.threshold:.2f}, {len(self.scored)}/{len(self.scores)} scored)"
        ]
        if not self.scored:
            lines.append("  No question could be scored -- treating that as a failure.")
        for score in self.worst:
            if score.faithfulness < self.threshold:
                lines.append(f"  {score.faithfulness:.3f}  {score.question[:70]}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def answer_questions(engine, questions: list[str]) -> list:
    """Ask the live pipeline every question, keeping answers and contexts."""
    answers = []
    for index, question in enumerate(questions, start=1):
        print(f"  [{index}/{len(questions)}] {question[:60]}")
        answers.append(engine.ask(question, session_id=f"ragas-{index}"))
    return answers


def score_faithfulness(questions: list[str], answers: list, llm=None) -> list[float | None]:
    """Run RAGAS faithfulness over the answered questions.

    Returns one score per question, ``None`` where RAGAS could not produce one
    (typically because nothing was retrieved, so there is nothing to be
    faithful to).
    """
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness

    samples, evaluated_indices = [], []
    for index, (question, answer) in enumerate(zip(questions, answers, strict=True)):
        if not answer.contexts:
            continue
        samples.append(
            SingleTurnSample(
                user_input=question,
                response=answer.answer,
                retrieved_contexts=list(answer.contexts),
            )
        )
        evaluated_indices.append(index)

    scores: list[float | None] = [None] * len(questions)
    if not samples:
        return scores

    result = evaluate(
        EvaluationDataset(samples=samples),
        metrics=[Faithfulness()],
        llm=LangchainLLMWrapper(llm) if llm is not None else None,
    )

    for position, index in enumerate(evaluated_indices):
        value = result.scores[position].get("faithfulness")
        # RAGAS returns NaN when a sample could not be judged.
        scores[index] = None if value is None or value != value else float(value)
    return scores


def evaluate_faithfulness(
    engine,
    questions: list[str],
    threshold: float = DEFAULT_MIN_FAITHFULNESS,
) -> FaithfulnessReport:
    """Answer every question with the live pipeline, then judge faithfulness."""
    print(f"Answering {len(questions)} questions...")
    answers = answer_questions(engine, questions)

    print("Scoring faithfulness with RAGAS...")
    scores = score_faithfulness(questions, answers, llm=engine.llm)

    return FaithfulnessReport(
        threshold=threshold,
        scores=[
            QuestionScore(question=question, faithfulness=score, sources=answer.sources)
            for question, answer, score in zip(questions, answers, scores, strict=True)
        ],
    )
