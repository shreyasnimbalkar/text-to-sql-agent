"""Safe, read-only access to a SQLite database.

The agent runs SQL that a language model wrote, so this module is the safety
boundary. Three layers of protection:

  1. The file is opened in read-only mode (`mode=ro`). SQLite itself refuses
     INSERT / UPDATE / DELETE / DROP, whatever the model writes.
  2. Only queries that start with SELECT or WITH are accepted, and SQLite's
     `execute` runs at most one statement, so "SELECT 1; DROP TABLE x" fails.
  3. Every query has a time limit, so a runaway query (e.g. an accidental
     cross join of two big tables) is cancelled instead of hanging forever.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

_READ_ONLY_START = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
_LEADING_COMMENTS = re.compile(r"^\s*(--[^\n]*\n|/\*.*?\*/)\s*", re.DOTALL)


@dataclass
class QueryResult:
    sql: str
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    error: str | None = None
    truncated: bool = False  # True if there were more rows than we fetched
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def is_empty(self) -> bool:
        """No rows, or only rows where every value is NULL (e.g. MAX() over nothing)."""
        return self.ok and all(all(v is None for v in row) for row in self.rows)

    def preview(self, max_rows: int = 10, max_chars: int = 1500) -> str:
        """A short plain-text table for showing the result to the model."""
        if self.error:
            return f"ERROR: {self.error}"
        if not self.rows:
            return f"(0 rows) columns: {', '.join(self.columns)}"
        lines = [" | ".join(self.columns)]
        for row in self.rows[:max_rows]:
            lines.append(" | ".join(_short(v) for v in row))
        more = len(self.rows) - max_rows
        if more > 0 or self.truncated:
            lines.append(f"... ({'more than ' if self.truncated else ''}{len(self.rows)} rows total)")
        text = "\n".join(lines)
        return text if len(text) <= max_chars else text[:max_chars] + "\n... (cut)"


def _short(v, limit: int = 60) -> str:
    s = "NULL" if v is None else str(v)
    return s if len(s) <= limit else s[: limit - 3] + "..."


def strip_sql(sql: str) -> str:
    """Remove leading comments and a trailing semicolon."""
    sql = sql.strip()
    while True:
        new = _LEADING_COMMENTS.sub("", sql, count=1)
        if new == sql:
            break
        sql = new
    return sql.rstrip().rstrip(";").rstrip()


class Database:
    def __init__(self, path: str | Path, timeout_s: float = 30.0):
        self.path = Path(path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.name = self.path.stem
        self.timeout_s = timeout_s
        # as_uri() turns spaces into %20 etc., which SQLite's URI parser understands.
        self.conn = sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True, check_same_thread=False)
        # Some BIRD databases contain bytes that are not valid UTF-8.
        self.conn.text_factory = lambda b: b.decode("utf-8", errors="replace")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def run(self, sql: str, max_rows: int | None = 1000, timeout_s: float | None = None) -> QueryResult:
        """Run one read-only query. Never raises for SQL problems: errors come back in `.error`."""
        sql = strip_sql(sql)
        result = QueryResult(sql=sql)
        if not _READ_ONLY_START.match(sql):
            result.error = "Only read-only queries (SELECT / WITH) are allowed."
            return result

        deadline = time.perf_counter() + (timeout_s or self.timeout_s)
        # SQLite calls this every N virtual-machine steps; returning True aborts the query.
        self.conn.set_progress_handler(lambda: time.perf_counter() > deadline, 10_000)
        started = time.perf_counter()
        try:
            cur = self.conn.execute(sql)
            result.columns = [d[0] for d in cur.description or []]
            if max_rows is None:
                result.rows = cur.fetchall()
            else:
                result.rows = cur.fetchmany(max_rows + 1)
                if len(result.rows) > max_rows:
                    result.rows = result.rows[:max_rows]
                    result.truncated = True
        except sqlite3.Error as e:
            msg = str(e)
            if msg == "interrupted":
                msg = f"Query timed out after {timeout_s or self.timeout_s:.0f}s (too slow - check joins and filters)."
            result.error = msg
        except Exception as e:  # e.g. a Python-level error while converting a value
            result.error = f"{type(e).__name__}: {e}"
        finally:
            self.conn.set_progress_handler(None, 0)
            result.elapsed_ms = (time.perf_counter() - started) * 1000
        return result
