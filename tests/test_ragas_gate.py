import json

import pytest

from rag_service.evaluation.ragas_eval import (
    DEFAULT_MIN_FAITHFULNESS,
    FaithfulnessReport,
    QuestionScore,
    load_questions,
)


def report(threshold=0.75, *scores):
    return FaithfulnessReport(
        threshold=threshold,
        scores=[QuestionScore(question=f"q{i}", faithfulness=s) for i, s in enumerate(scores)],
    )


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #
def test_default_threshold_is_the_documented_one():
    assert DEFAULT_MIN_FAITHFULNESS == 0.75


def test_passes_above_the_threshold():
    r = report(0.75, 0.9, 0.8, 0.85)

    assert r.mean == pytest.approx(0.85)
    assert r.passed


def test_fails_below_the_threshold():
    r = report(0.75, 0.9, 0.5, 0.6)

    assert r.mean == pytest.approx(2.0 / 3)
    assert not r.passed


def test_exactly_at_the_threshold_passes():
    assert report(0.75, 0.75, 0.75).passed


def test_a_single_bad_answer_can_sink_the_mean():
    assert not report(0.75, 1.0, 1.0, 0.0).passed


# --------------------------------------------------------------------------- #
# Unscored questions
# --------------------------------------------------------------------------- #
def test_unscored_questions_are_excluded_from_the_mean():
    """An honest "not in the documents" reply retrieves nothing and cannot be
    scored; averaging it in as zero would punish the right behaviour."""
    r = report(0.75, 0.9, None, 0.9)

    assert r.mean == pytest.approx(0.9)
    assert r.unscored == ["q1"]
    assert r.passed


def test_nothing_scored_is_a_failure_not_a_pass():
    """An empty score list must never pass the gate by vacuous truth."""
    r = report(0.75, None, None)

    assert r.scored == []
    assert r.mean == 0.0
    assert not r.passed
    assert "treating that as a failure" in r.summary()


def test_report_with_no_questions_at_all_fails():
    assert not FaithfulnessReport(threshold=0.75, scores=[]).passed


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def test_summary_names_the_failing_questions():
    r = report(0.75, 0.2, 0.95)

    summary = r.summary()

    assert "FAIL" in summary
    assert "q0" in summary
    assert "q1" not in summary  # above threshold, not worth listing


def test_summary_says_pass_when_it_passes():
    assert report(0.75, 0.9).summary().startswith("PASS")


def test_worst_lists_lowest_first_and_caps_at_five():
    r = report(0.75, 0.9, 0.1, 0.5, 0.2, 0.3, 0.4, 0.45)

    worst = r.worst

    assert len(worst) == 5
    assert [s.faithfulness for s in worst] == [0.1, 0.2, 0.3, 0.4, 0.45]


def test_report_round_trips_to_disk(tmp_path):
    r = report(0.8, 0.9, None)

    path = r.write(tmp_path / "nested" / "ragas.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["threshold"] == 0.8
    assert payload["mean_faithfulness"] == pytest.approx(0.9)
    assert payload["questions"] == 2
    assert payload["scored"] == 1
    assert payload["passed"] is True


# --------------------------------------------------------------------------- #
# Question set
# --------------------------------------------------------------------------- #
def test_loads_plain_string_questions(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(json.dumps(["What is X?", "What is Y?"]), encoding="utf-8")

    assert load_questions(path) == ["What is X?", "What is Y?"]


def test_loads_object_questions(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(json.dumps([{"question": "What is X?"}]), encoding="utf-8")

    assert load_questions(path) == ["What is X?"]


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ([{"question": "  "}], "empty question"),
        ([""], "empty question"),
        ([], "no questions"),
        ({"question": "x"}, "JSON array"),
    ],
    ids=["blank", "empty-string", "empty-array", "not-an-array"],
)
def test_rejects_malformed_question_sets(tmp_path, payload, match):
    path = tmp_path / "q.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_questions(path)


def test_the_committed_question_set_meets_the_ci_requirement():
    """CI gates on a 20-question set; keep it from silently shrinking."""
    from rag_service.config import PROJECT_ROOT

    questions = load_questions(PROJECT_ROOT / "data" / "ragas_questions.json")

    assert len(questions) >= 20
    assert len(set(questions)) == len(questions), "duplicate questions"


# --------------------------------------------------------------------------- #
# The CLI exit code -- this is what actually fails the CI job
# --------------------------------------------------------------------------- #
def run_cli(monkeypatch, tmp_path, scores, threshold="0.75"):
    """Invoke rag-eval with the engine and the RAGAS call stubbed out."""
    import rag_service.evaluation.ragas_eval as ragas_eval
    import rag_service.rag as rag

    questions_path = tmp_path / "q.json"
    questions_path.write_text(
        json.dumps([f"question {i}" for i in range(len(scores))]), encoding="utf-8"
    )
    report_path = tmp_path / "ragas.json"

    class StubAnswer:
        answer = "an answer"
        sources = ["Article 1"]
        contexts = ["a passage"]

    class StubEngine:
        llm = None

    monkeypatch.setattr(rag, "RagEngine", lambda *a, **k: StubEngine())
    monkeypatch.setattr(
        ragas_eval, "answer_questions", lambda engine, qs: [StubAnswer() for _ in qs]
    )
    monkeypatch.setattr(ragas_eval, "score_faithfulness", lambda qs, ans, llm=None: scores)
    monkeypatch.setattr(
        "sys.argv",
        [
            "rag-eval",
            "--questions",
            str(questions_path),
            "--min-faithfulness",
            threshold,
            "--report",
            str(report_path),
        ],
    )

    from rag_service.cli import evaluate_faithfulness_cli

    try:
        evaluate_faithfulness_cli()
        code = 0
    except SystemExit as exc:
        code = exc.code
    return code, report_path


def test_cli_exits_zero_when_the_gate_passes(monkeypatch, tmp_path):
    code, report_path = run_cli(monkeypatch, tmp_path, [0.9, 0.85])

    assert code == 0
    assert json.loads(report_path.read_text(encoding="utf-8"))["passed"] is True


def test_cli_exits_nonzero_when_faithfulness_is_too_low(monkeypatch, tmp_path):
    """CI must go red below the threshold."""
    code, report_path = run_cli(monkeypatch, tmp_path, [0.9, 0.2])

    assert code == 1
    assert json.loads(report_path.read_text(encoding="utf-8"))["passed"] is False


def test_cli_writes_the_report_even_on_failure(monkeypatch, tmp_path):
    _, report_path = run_cli(monkeypatch, tmp_path, [0.1])

    assert report_path.exists(), "the report must survive a failing gate for debugging"


def test_cli_honours_a_custom_threshold(monkeypatch, tmp_path):
    passing, _ = run_cli(monkeypatch, tmp_path, [0.5], threshold="0.4")
    failing, _ = run_cli(monkeypatch, tmp_path, [0.5], threshold="0.9")

    assert passing == 0
    assert failing == 1
