"""The agent loop: explore -> write SQL -> run it -> fix it if something is wrong.

    question
       |
       v
  [schema + example values]  (automatic exploration, done once per database)
       |
       v
    model --EXPLORE sql--> run it, show the model the rows --> back to model   (up to N times)
       |
     ANSWER sql
       |
       v
    run it ---- ok, has rows ------------------------------> done
       |
       +-- error / no rows --> tell the model what went wrong --> back to model  (up to M fixes)

The first ANSWER the model gives is recorded separately (`first_sql`). That is
exactly what the agent would have returned with self-correction switched off,
so a single benchmark run measures both "with" and "without" self-correction.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from . import prompts
from .db import Database, QueryResult, strip_sql
from .llm import LLM

_SQL_BLOCK = re.compile(r"```(?:sql|sqlite)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_BARE_SELECT = re.compile(r"\b((?:SELECT|WITH)\b.*)", re.DOTALL | re.IGNORECASE)


@dataclass
class Step:
    kind: str  # "explore" | "answer" | "invalid"
    sql: str | None
    result: QueryResult | None = None
    feedback: str | None = None  # what we told the model after this step (error, empty result, ...)


@dataclass
class AgentResult:
    question: str
    sql: str | None  # final answer
    result: QueryResult | None
    first_sql: str | None  # first ANSWER = the answer without self-correction
    first_result: QueryResult | None
    steps: list[Step] = field(default_factory=list)
    fixes: int = 0  # how many times we sent the model back to fix its answer
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0

    @property
    def explores(self) -> int:
        return sum(s.kind == "explore" for s in self.steps)


def parse_reply(text: str) -> tuple[str, str | None]:
    """Find the action (EXPLORE or ANSWER) and the SQL in a model reply."""
    blocks = [b.strip() for b in _SQL_BLOCK.findall(text) if b.strip()]
    if blocks:
        sql = blocks[-1]
        before = text[: text.rfind(blocks[-1])]
    else:
        m = _BARE_SELECT.search(text)
        sql = m.group(1).strip() if m else None
        before = text[: m.start()] if m else text
    # Look at the last action word written before the query. Default to ANSWER.
    actions = re.findall(r"\b(EXPLORE|ANSWER)\b", before)
    action = "explore" if actions and actions[-1] == "EXPLORE" else "answer"
    return action, (strip_sql(sql) if sql else None)


def _problem(result: QueryResult) -> str | None:
    if not result.ok:
        return prompts.FIX_ERROR.format(error=result.error)
    if result.is_empty or _all_zero(result):
        return prompts.FIX_EMPTY.format(preview=result.preview(max_rows=5))
    return None


def _all_zero(result: QueryResult) -> bool:
    """One row of only 0 / NULL, e.g. COUNT(*) = 0. Usually a filter that matched nothing."""
    return len(result.rows) == 1 and all(v is None or v == 0 for v in result.rows[0])


class Agent:
    def __init__(
        self,
        llm: LLM,
        db: Database,
        schema_text: str,
        *,
        max_explore: int = 3,
        max_fixes: int = 2,
        self_correct: bool = True,
        max_steps: int = 8,
    ):
        self.llm, self.db = llm, db
        self.max_explore = max_explore
        self.max_fixes = max_fixes if self_correct else 0
        self.max_steps = max_steps
        self.system = prompts.SYSTEM.format(schema=schema_text, max_explore=max_explore)

    def ask(self, question: str, hint: str = "", on_step=None) -> AgentResult:
        """Answer one question. `on_step(step)` is called after every step (for live display)."""
        started = time.perf_counter()
        usage_before = (self.llm.usage.calls, self.llm.usage.prompt_tokens, self.llm.usage.completion_tokens)
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": prompts.QUESTION.format(question=question, hint=hint or "(none)")},
        ]
        out = AgentResult(question=question, sql=None, result=None, first_sql=None, first_result=None)
        explores = 0
        last_sql = None

        for _ in range(self.max_steps):
            reply = self.llm.chat(messages)
            action, sql = parse_reply(reply)

            if sql is None:
                step = Step("invalid", None, feedback=prompts.NO_SQL_FOUND)
                messages += [{"role": "assistant", "content": reply[-1500:]}, {"role": "user", "content": step.feedback}]
                out.steps.append(step)
                _notify(on_step, step)
                continue

            # Keep the history short: store just the action and query, not the model's explanation.
            messages.append({"role": "assistant", "content": f"{action.upper()}\n```sql\n{sql}\n```"})
            last_sql = sql

            if action == "explore":
                if explores >= self.max_explore:
                    step = Step("explore", sql, feedback=prompts.NO_EXPLORES_LEFT)
                else:
                    explores += 1
                    res = self.db.run(sql, max_rows=50)
                    step = Step("explore", sql, res)
                    step.feedback = prompts.EXPLORE_RESULT.format(preview=res.preview(max_rows=20), left=self.max_explore - explores)
                messages.append({"role": "user", "content": step.feedback})
                out.steps.append(step)
                _notify(on_step, step)
                continue

            # ANSWER: run it.
            res = self.db.run(sql)
            step = Step("answer", sql, res)
            out.steps.append(step)
            resubmitted = out.sql is not None and sql == out.sql
            if out.first_sql is None:
                out.first_sql, out.first_result = sql, res
            out.sql, out.result = sql, res

            problem = _problem(res)
            # Stop if the answer looks fine, we are out of fixes, or the model insists an empty result is right.
            if problem is None or out.fixes >= self.max_fixes or (resubmitted and res.ok):
                _notify(on_step, step)
                break
            out.fixes += 1
            step.feedback = problem
            messages.append({"role": "user", "content": problem})
            _notify(on_step, step)

        if out.sql is None and last_sql:
            # The model never gave an ANSWER (it only explored). Use its last query rather than nothing.
            out.sql = out.first_sql = last_sql
            out.result = out.first_result = self.db.run(last_sql)

        out.llm_calls = self.llm.usage.calls - usage_before[0]
        out.prompt_tokens = self.llm.usage.prompt_tokens - usage_before[1]
        out.completion_tokens = self.llm.usage.completion_tokens - usage_before[2]
        out.seconds = time.perf_counter() - started
        return out


def _notify(cb, step: Step) -> None:
    if cb is not None:
        cb(step)
