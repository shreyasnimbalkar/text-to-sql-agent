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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..agent import Agent
from ..db import Database, QueryResult
from ..llm import LLM, LLMError
from ..schema import describe
from . import bird

SCORE_TIMEOUT_S = 60  # BIRD's evaluation script also gives each query a time budget
MAX_FAILURE_STREAK = 5  # this many skipped questions in a row = the provider is down, stop


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
    workers: int = 1,
    timeout_s: float | None = None,
) -> None:
    """Answer and score every question.

    With workers > 1, several questions run at once, each worker with its own model
    client and database connections. A question whose model call times out is
    recorded as skipped and retried on the next run; a daily limit, a bad key, or
    MAX_FAILURE_STREAK skips in a row stop the run.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _done_ids(out)
    todo = [q for q in questions if q.question_id not in done]
    print(f"{len(questions)} questions, {len(done & {q.question_id for q in questions})} already done, "
          f"{len(todo)} to go -> {out}", flush=True)

    lock = threading.Lock()
    stop = threading.Event()
    local = threading.local()
    schemas: dict[str, str] = {}
    open_dbs: list[Database] = []
    tally = {"started": 0, "n": 0, "first": 0, "final": 0, "skipped": 0, "streak": 0}

    def database(db_id: str) -> Database:
        if not hasattr(local, "dbs"):
            local.dbs = {}
        if db_id not in local.dbs:
            db = Database(bird.db_path(db_id))
            with lock:
                open_dbs.append(db)
            local.dbs[db_id] = db
            if db_id not in schemas:
                descs = bird.load_descriptions(db_id) if use_descriptions else None
                schemas[db_id] = describe(db, descs)
        return local.dbs[db_id]

    def work(q: bird.Question) -> None:
        if stop.is_set():
            return
        if not hasattr(local, "llm"):
            local.llm = LLM(model, extra=extra or {}, timeout_s=timeout_s)
        db = database(q.db_id)
        agent = Agent(local.llm, db, schemas[q.db_id], max_explore=max_explore, max_fixes=max_fixes,
                      self_correct=self_correct)
        with lock:
            tally["started"] += 1

        rec = {"question_id": q.question_id, "db_id": q.db_id, "difficulty": q.difficulty, "model": model,
               "self_correct": self_correct, "question": q.question, "evidence": q.evidence, "gold_sql": q.gold_sql}
        try:
            res = agent.ask(q.question, q.evidence)
        except LLMError as e:
            rec["llm_error"] = str(e)
            with lock:
                _append(out, rec)
                tally["skipped"] += 1
                tally["streak"] += 1
                print(f"skipped q{q.question_id}: {str(e)[:120]}", flush=True)
                if not e.transient or tally["streak"] >= MAX_FAILURE_STREAK:
                    if not stop.is_set():
                        print(f"\nStopped: {e}\nRun the same command again later to resume.", flush=True)
                    stop.set()
            return


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
        mark = "OK " if rec["correct_final"] else "-- "
        fixed = " (fixed by self-correction)" if rec["correct_final"] and not rec["correct_first"] else ""
        with lock:
            _append(out, rec)
            tally["streak"] = 0
            tally["n"] += 1
            tally["first"] += rec["correct_first"]
            tally["final"] += rec["correct_final"]
            n = tally["n"]
            print(f"[{n}/{len(todo)}] {mark} q{q.question_id} {q.db_id} {q.difficulty:<11} "
                  f"{res.llm_calls} calls {res.seconds:5.1f}s{fixed}   "
                  f"| this session: {100 * tally['final'] / n:.1f}% (first try {100 * tally['first'] / n:.1f}%)",
                  flush=True)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(work, todo))
    for db in open_dbs:
        db.close()
    if tally["skipped"]:
        print(f"\n{tally['skipped']} question(s) skipped after model errors. "
              f"Run the same command again to retry them.", flush=True)


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
