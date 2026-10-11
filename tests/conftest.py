import sqlite3

import pytest

from sqlagent.db import Database


@pytest.fixture
def db_path(tmp_path):
    """A tiny shop database: customers and their orders (note the table named "order")."""
    path = tmp_path / "shop.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, "Home Country" TEXT);
        CREATE TABLE "order" (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER REFERENCES customers(id),
            amount REAL
        );
        INSERT INTO customers VALUES (1, 'Ana', 'Spain'), (2, 'Ben', 'France'), (3, 'Chloe', 'Spain');
        INSERT INTO "order" VALUES (10, 1, 20.0), (11, 1, 5.5), (12, 2, 99.0);
        """
    )
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def db(db_path):
    with Database(db_path) as d:
        yield d


class ScriptedLLM:
    """Stands in for a real model: returns pre-written replies in order and records what it was sent."""

    def __init__(self, replies):
        from sqlagent.llm import Usage

        self.replies = list(replies)
        self.sent = []
        self.usage = Usage()

    def chat(self, messages):
        self.sent.append([dict(m) for m in messages])
        self.usage.calls += 1
        self.usage.prompt_tokens += 100
        self.usage.completion_tokens += 10
        return self.replies.pop(0)
