"""Run the agent over BIRD questions and score every answer.

Scoring is BIRD's official "execution accuracy" (EX): run the predicted SQL and
the gold SQL on the same database; the answer counts as correct when both
return the same SET of rows. Column names, row order and duplicate rows do not
matter; the values do.

Each question becomes one line in a JSONL file. If the run stops (Ctrl+C,
daily rate limit), running the same command again skips finished questions.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..agent import Agent
from ..db import Database, QueryResult
from ..llm import LLM, LLMError
from ..schema import describe
from . import bird

SCORE_TIMEOUT_S = 60  # BIRD's evaluation script also gives each query a time budget


def same_result(pred: QueryResult | None, gold: QueryResult) -> bool:
    """BIRD EX: correct iff set(predicted rows) == set(gold rows)."""
    if pred is None or not pred.ok or not gold.ok:
        return False
    return set(pred.rows) == set(gold.rows)


def _done_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            if not rec.get("llm_error"):
                done.add(rec["question_id"])
    return done


def run(
    model: str,
    out: Path,
    questions: list[bird.Question],
    *,
    self_correct: bool = True,
    max_explore: int = 3,
    max_fixes: int = 2,
    use_descriptions: bool = True,
    extra: dict | None = None,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _done_ids(out)
    todo = [q for q in questions if q.question_id not in done]
    print(f"{len(questions)} questions, {len(done & {q.question_id for q in questions})} already done, {len(todo)} to go -> {out}")

    llm = LLM(model, extra=extra or {})
    dbs: dict[str, Database] = {}
    schemas: dict[str, str] = {}
    n = n_first = n_final = 0

    for i, q in enumerate(todo, 1):
        if q.db_id not in dbs:
            dbs[q.db_id] = Database(bird.db_path(q.db_id))
            descs = bird.load_descriptions(q.db_id) if use_descriptions else None
            schemas[q.db_id] = describe(dbs[q.db_id], descs)
        db = dbs[q.db_id]
        agent = Agent(llm, db, schemas[q.db_id], max_explore=max_explore, max_fixes=max_fixes, self_correct=self_correct)

        rec = {"question_id": q.question_id, "db_id": q.db_id, "difficulty": q.difficulty, "model": model,
               "self_correct": self_correct, "question": q.question, "evidence": q.evidence, "gold_sql": q.gold_sql}
        try:
            res = agent.ask(q.question, q.evidence)
        except LLMError as e:
            rec["llm_error"] = str(e)
            _append(out, rec)
            print(f"\nStopped: {e}\nRun the same command again later to resume.")
            break

        # Score with full results (no row limit), like the official evaluation.
        gold = db.run(q.gold_sql, max_rows=None, timeout_s=SCORE_TIMEOUT_S)
        final = db.run(res.sql, max_rows=None, timeout_s=SCORE_TIMEOUT_S) if res.sql else None
        first = final if res.first_sql == res.sql else (
            db.run(res.first_sql, max_rows=None, timeout_s=SCORE_TIMEOUT_S) if res.first_sql else None)

        rec.update(
            first_sql=res.first_sql,
            final_sql=res.sql,
            first_error=first.error if first else "no SQL",
            final_error=final.error if final else "no SQL",
            correct_first=same_result(first, gold),
            correct_final=same_result(final, gold),
            gold_error=gold.error,
            fixes=res.fixes,
            explores=res.explores,
            llm_calls=res.llm_calls,
            prompt_tokens=res.prompt_tokens,
            completion_tokens=res.completion_tokens,
            seconds=round(res.seconds, 2),
            steps=[{"kind": s.kind, "sql": s.sql, "error": s.result.error if s.result else None,
                    "rows": len(s.result.rows) if s.result and s.result.ok else None} for s in res.steps],
        )
        _append(out, rec)

        n += 1
        n_first += rec["correct_first"]
        n_final += rec["correct_final"]
        mark = "OK " if rec["correct_final"] else "-- "
        fixed = " (fixed by self-correction)" if rec["correct_final"] and not rec["correct_first"] else ""
        print(f"[{i}/{len(todo)}] {mark} q{q.question_id} {q.db_id} {q.difficulty:<11} "
              f"{res.llm_calls} calls {res.seconds:5.1f}s{fixed}   "
              f"| this session: {100 * n_final / n:.1f}% (first try {100 * n_first / n:.1f}%)", flush=True)

    for db in dbs.values():
        db.close()


def _append(path: Path, rec: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def gold_check(questions: list[bird.Question]) -> None:
    """Sanity check: every gold query must run on our copy of the databases."""
    bad = 0
    started = time.perf_counter()
    dbs: dict[str, Database] = {}
    for q in questions:
        db = dbs.setdefault(q.db_id, Database(bird.db_path(q.db_id)))
        r = db.run(q.gold_sql, max_rows=None, timeout_s=SCORE_TIMEOUT_S)
        if not r.ok:
            bad += 1
            print(f"q{q.question_id} {q.db_id}: {r.error}")
        elif r.is_empty:
            print(f"q{q.question_id} {q.db_id}: gold result is empty")
    print(f"{len(questions) - bad}/{len(questions)} gold queries run OK ({time.perf_counter() - started:.0f}s)")
