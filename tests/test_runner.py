from sqlagent.bench import bird, report, run
from sqlagent.llm import LLMError, Usage

ANSWERS = {
    "How many customers?": "SELECT COUNT(*) FROM customers",
    "Who lives in France?": "SELECT name FROM customers WHERE \"Home Country\" = 'France'",
    "Biggest order?": "SELECT MAX(amount) FROM \"order\"",
}


class FakeLLM:
    """Answers from ANSWERS; questions listed in `fail` raise the given error instead."""

    fail: dict[str, LLMError] = {}

    def __init__(self, model, extra=None):
        self.usage = Usage()

    def chat(self, messages):
        question = messages[1]["content"].splitlines()[0].removeprefix("Question: ")
        if question in self.fail:
            raise self.fail[question]
        self.usage.calls += 1
        return f"ANSWER\n```sql\n{ANSWERS[question]}\n```"


def setup(monkeypatch, db_path, fail):
    FakeLLM.fail = fail
    monkeypatch.setattr(run, "LLM", FakeLLM)
    monkeypatch.setattr(bird, "db_path", lambda db_id: db_path)
    monkeypatch.setattr(bird, "load_descriptions", lambda db_id: {})
    return [bird.Question(i, "shop", q, "", sql, "simple") for i, (q, sql) in enumerate(ANSWERS.items())]


def test_parallel_run_skips_timeouts_and_retries_them(monkeypatch, db_path, tmp_path):
    out = tmp_path / "run.jsonl"
    questions = setup(monkeypatch, db_path, {"Who lives in France?": LLMError("timeout", transient=True)})
    run.run("fake:model", out, questions, workers=2)

    recs = report.load(out)  # skipped questions are not counted
    assert sorted(r["question_id"] for r in recs) == [0, 2]
    assert all(r["correct_final"] for r in recs)

    FakeLLM.fail = {}
    run.run("fake:model", out, questions, workers=2)  # second run only does the skipped one
    recs = report.load(out)
    assert sorted(r["question_id"] for r in recs) == [0, 1, 2]
    assert len(out.read_text(encoding="utf-8").splitlines()) == 4  # 2 + 1 skipped + 1 retry


def test_fatal_error_stops_the_run(monkeypatch, db_path, tmp_path):
    out = tmp_path / "run.jsonl"
    fatal = LLMError("daily limit", transient=False)
    questions = setup(monkeypatch, db_path, {q: fatal for q in ANSWERS})
    run.run("fake:model", out, questions, workers=1)
    assert len(out.read_text(encoding="utf-8").splitlines()) == 1  # stopped after the first failure
