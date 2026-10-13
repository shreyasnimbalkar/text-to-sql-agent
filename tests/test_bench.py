import json

from sqlagent.bench import bird, report
from sqlagent.bench.run import same_result
from sqlagent.db import QueryResult


def qr(rows, error=None):
    return QueryResult(sql="", rows=rows, error=error)


def test_execution_accuracy_rules():
    gold = qr([("a", 1), ("b", 2)])
    assert same_result(qr([("b", 2), ("a", 1)]), gold)  # order does not matter
    assert same_result(qr([("a", 1), ("a", 1), ("b", 2)]), gold)  # duplicates do not matter (set)
    assert not same_result(qr([("a", 1)]), gold)  # missing a row
    assert not same_result(qr([("a", 1, "x"), ("b", 2, "y")]), gold)  # extra column
    assert not same_result(qr([], error="boom"), gold)
    assert not same_result(None, gold)


def test_failure_breakdown(monkeypatch, db_path):
    monkeypatch.setattr(bird, "db_path", lambda db_id: db_path)
    gold = "SELECT name FROM customers WHERE \"Home Country\" = 'Spain'"
    cases = {
        "SELECT nope FROM customers": "query fails (SQL error or timeout)",
        "SELECT name FROM customers WHERE id = 99": "query returns no data",
        "SELECT name, id FROM customers WHERE \"Home Country\" = 'Spain'": "wrong columns (extra, missing or merged)",
        "SELECT name FROM customers WHERE id = 1": "too many or too few rows (DISTINCT, LIMIT, ties)",
        "SELECT name FROM customers WHERE \"Home Country\" = 'France'": "wrong values (filter, join or calculation)",
    }
    recs = [{"correct_final": False, "db_id": "shop", "gold_sql": gold, "final_sql": sql} for sql in cases]
    recs.append({"correct_final": False, "db_id": "shop", "gold_sql": "SELECT 1.0 / 3", "final_sql": "SELECT 0.333333"})
    got = report.failure_breakdown(recs)
    assert got == {**{why: 1 for why in cases.values()}, "same numbers, different rounding": 1}


def test_stratified_sample_keeps_difficulty_mix():
    qs = [bird.Question(i, "db", "q", "", "SELECT 1", d)
          for i, d in enumerate(["simple"] * 30 + ["moderate"] * 50 + ["challenging"] * 20)]
    s = bird.sample(qs, 10, seed=1)
    assert [q.difficulty for q in s].count("moderate") == 5
    assert [q.difficulty for q in s].count("challenging") == 2
    assert s == bird.sample(qs, 10, seed=1)  # deterministic


def test_description_csv_encodings(tmp_path):
    folder = tmp_path / "database_description"
    folder.mkdir()
    header = "original_column_name,column_name,column_description,data_format,value_description\n"
    (folder / "people.csv").write_bytes(
        ("﻿" + header + "pid,person id,,integer,\n"
         "dob,,date of birth,text,commonsense evidence: format YYYY-MM-DD\n"
         "name,name,name,text,\n").encode("utf-8"))
    (folder / "cities.csv").write_bytes((header + "nm,city name,Name of the caf\xe9,text,\n").encode("cp1252"))
    d = bird.read_description_folder(folder)
    assert d["people"]["pid"] == "person id"
    assert d["people"]["dob"] == "date of birth; format YYYY-MM-DD"
    assert "name" not in d["people"]  # description adds nothing -> skipped
    assert d["cities"]["nm"] == "city name; Name of the café"


def test_report(tmp_path):
    base = {"model": "x:y", "first_error": None, "final_error": None, "fixes": 0, "explores": 0,
            "llm_calls": 1, "prompt_tokens": 100, "completion_tokens": 10, "seconds": 1.0}
    rows = [
        {**base, "question_id": 1, "difficulty": "simple", "correct_first": True, "correct_final": True},
        {**base, "question_id": 2, "difficulty": "simple", "correct_first": False, "correct_final": True,
         "fixes": 1, "first_error": "no such column"},
        {**base, "question_id": 3, "difficulty": "moderate", "correct_first": False, "correct_final": False},
        {**base, "question_id": 4, "difficulty": "challenging", "correct_first": True, "correct_final": False,
         "fixes": 1},
        {"question_id": 5, "llm_error": "rate limited"},  # unfinished -> ignored
    ]
    path = tmp_path / "run.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    s = report.summarize(report.load(path))
    assert s["n"] == 4 and s["ex_first"] == 0.5 and s["ex_final"] == 0.5
    assert s["fixed"] == 1 and s["broke"] == 1 and s["fix_attempted"] == 2
    text = report.render([path])
    assert "fixed 1" in text and "+0.0 points" in text
