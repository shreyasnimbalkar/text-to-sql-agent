"""Loading the BIRD benchmark: questions, databases, and column descriptions.

Folder layout expected under data/bird/ (created by scripts/download_bird.py):

  mini_dev_sqlite.json                 500 questions (BIRD Mini-Dev, SQLite version)
  dev.json                             1,534 questions (full BIRD dev set)
  dev_databases/<db_id>/<db_id>.sqlite
  dev_databases/<db_id>/database_description/<table>.csv
"""

from __future__ import annotations

import csv
import io
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from ..schema import Descriptions

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "bird"
DIFFICULTIES = ("simple", "moderate", "challenging")


@dataclass
class Question:
    question_id: int
    db_id: str
    question: str
    evidence: str  # BIRD's "external knowledge" hint, e.g. "Eligible free rate = Free Meal Count / Enrollment"
    gold_sql: str
    difficulty: str


def load_questions(path: str | Path | None = None) -> list[Question]:
    path = Path(path) if path else DATA_DIR / "mini_dev_sqlite.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for i, q in enumerate(raw):
        out.append(
            Question(
                question_id=int(q.get("question_id", i)),
                db_id=q["db_id"],
                question=q["question"],
                evidence=q.get("evidence", "") or "",
                gold_sql=q.get("SQL") or q.get("sql") or "",
                difficulty=q.get("difficulty", "unknown"),
            )
        )
    return out


def sample(questions: list[Question], n: int, seed: int = 0) -> list[Question]:
    """A deterministic sample that keeps the same mix of difficulties as the full set."""
    if n >= len(questions):
        return list(questions)
    rng = random.Random(seed)
    by_diff: dict[str, list[Question]] = {}
    for q in questions:
        by_diff.setdefault(q.difficulty, []).append(q)
    picked = []
    for qs in by_diff.values():
        k = round(n * len(qs) / len(questions))
        picked.extend(rng.sample(qs, min(k, len(qs))))
    picked.sort(key=lambda q: q.question_id)
    return picked[:n]


def db_path(db_id: str, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / "dev_databases" / db_id / f"{db_id}.sqlite"


def _read_text(path: Path) -> str:
    """BIRD's CSVs are a mix of UTF-8, UTF-8 with BOM, and Windows-1252."""
    data = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def load_descriptions(db_id: str, data_dir: Path = DATA_DIR) -> Descriptions:
    return read_description_folder(data_dir / "dev_databases" / db_id / "database_description")


def read_description_folder(folder: Path) -> Descriptions:
    """Read BIRD-style database_description/*.csv into {table: {column: description}}."""
    out: Descriptions = {}
    if not folder.exists():
        return out
    for f in sorted(folder.glob("*.csv")):
        cols: dict[str, str] = {}
        for row in csv.DictReader(io.StringIO(_read_text(f))):
            row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
            name = row.get("original_column_name", "")
            if not name:
                continue
            parts = []
            full_name = row.get("column_name", "")
            desc = row.get("column_description", "")
            if full_name and _norm(full_name) != _norm(name):
                parts.append(full_name)
            if desc and _norm(desc) not in (_norm(name), _norm(full_name)):
                parts.append(desc)
            if value_desc := row.get("value_description", ""):
                # Many BIRD descriptions start with this filler label; drop it to save tokens.
                value_desc = re.sub(r"^\s*commonsense (evidence|reasoning):\s*", "", value_desc, flags=re.IGNORECASE)
                parts.append(value_desc)
            if parts:
                cols[name.lower()] = "; ".join(parts)
        out[f.stem.lower()] = cols
    return out
