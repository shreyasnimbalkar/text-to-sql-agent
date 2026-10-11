"""Turn a database into a compact text description the model can read.

This is the agent's first "exploration" step, done automatically before the
model writes any SQL. For every table we collect:

  * columns, their types, and which column is the primary key
  * foreign keys (how tables join to each other)
  * the row count
  * a few real example values per column, so the model sees that dates look
    like '2012-03-01' or that a flag is 'Y'/'N' rather than 1/0
  * optional human-written column descriptions (BIRD ships these as CSV files)

The result looks like a CREATE TABLE statement with comments, which models
have seen millions of times in training data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .db import Database

# Descriptions keyed by lower-case table name, then lower-case column name.
Descriptions = dict[str, dict[str, str]]

_SAFE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# SQL keywords that real databases use as table/column names (BIRD has a table called "order").
_KEYWORDS = set(
    "abort action add all alter and as asc between by case cast check collate column commit constraint "
    "create cross current current_date current_time current_timestamp default delete desc distinct drop "
    "else end escape except exists filter foreign from full glob group having if in index inner insert "
    "intersect into is isnull join key left like limit match natural no not notnull null of offset on or "
    "order outer over primary references regexp replace right row rows select set table then to "
    "transaction union unique update using values view when where window with".split()
)


def quote(name: str) -> str:
    """Quote an identifier only when it needs it (spaces, punctuation, keywords)."""
    if _SAFE_IDENT.match(name) and name.lower() not in _KEYWORDS:
        return name
    return '"' + name.replace('"', '""') + '"'


@dataclass
class Column:
    name: str
    type: str
    pk: bool
    examples: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class Table:
    name: str
    row_count: int | None
    columns: list[Column]
    foreign_keys: list[tuple[str, str, str]]  # (from_column, to_table, to_column)


def inspect(db: Database, examples_per_column: int = 3, descriptions: Descriptions | None = None) -> list[Table]:
    descriptions = descriptions or {}
    names = db.run("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
    tables = []
    for (tname,) in names.rows:
        count = db.run(f"SELECT COUNT(*) FROM {quote(tname)}", timeout_s=5)
        info = db.conn.execute(f"PRAGMA table_info({quote(tname)})").fetchall()
        fks = db.conn.execute(f"PRAGMA foreign_key_list({quote(tname)})").fetchall()
        tdesc = descriptions.get(tname.lower(), {})
        columns = []
        for _cid, cname, ctype, _notnull, _default, pk in info:
            col = Column(name=cname, type=ctype or "", pk=bool(pk), description=tdesc.get(cname.lower(), ""))
            if examples_per_column:
                col.examples = _examples(db, tname, cname, examples_per_column)
            columns.append(col)
        tables.append(
            Table(
                name=tname,
                row_count=count.rows[0][0] if count.ok else None,
                columns=columns,
                # PRAGMA foreign_key_list: (id, seq, table, from, to, ...). `to` can be NULL = the PK.
                foreign_keys=[(fk[3], fk[2], fk[4] or "?") for fk in fks],
            )
        )
    return tables


def _examples(db: Database, table: str, column: str, n: int) -> list[str]:
    res = db.run(
        f"SELECT DISTINCT {quote(column)} FROM {quote(table)} WHERE {quote(column)} IS NOT NULL LIMIT {n}",
        timeout_s=2,
    )
    if not res.ok:
        return []
    out = []
    for (v,) in res.rows:
        if isinstance(v, (bytes, bytearray)):
            continue
        s = repr(v) if isinstance(v, str) else str(v)
        out.append(s if len(s) <= 40 else s[:37] + "...'")
    return out


def render(tables: list[Table], max_description_chars: int = 120) -> str:
    """Render tables as commented CREATE TABLE statements."""
    blocks = []
    for t in tables:
        rows = f"  -- {t.row_count} rows" if t.row_count is not None else ""
        lines = [f"CREATE TABLE {quote(t.name)} ({rows}"]
        body = []
        for c in t.columns:
            line = f"  {quote(c.name)} {c.type}".rstrip()
            if c.pk:
                line += " PRIMARY KEY"
            notes = []
            if c.description:
                d = " ".join(c.description.split())
                notes.append(d if len(d) <= max_description_chars else d[: max_description_chars - 3] + "...")
            if c.examples:
                notes.append("e.g. " + ", ".join(c.examples))
            body.append((line, " | ".join(notes)))
        for fk_from, fk_table, fk_to in t.foreign_keys:
            body.append((f"  FOREIGN KEY ({quote(fk_from)}) REFERENCES {quote(fk_table)}({quote(fk_to)})", ""))
        for i, (line, note) in enumerate(body):
            comma = "," if i < len(body) - 1 else ""
            lines.append(f"{line}{comma}" + (f"  -- {note}" if note else ""))
        lines.append(");")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def describe(db: Database, descriptions: Descriptions | None = None, examples_per_column: int = 3) -> str:
    """One call: inspect the database and render the schema text."""
    return render(inspect(db, examples_per_column=examples_per_column, descriptions=descriptions))
