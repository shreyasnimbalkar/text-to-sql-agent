from sqlagent.db import Database, strip_sql
from sqlagent.schema import describe, quote


def test_select_works(db):
    r = db.run('SELECT name FROM customers WHERE "Home Country" = \'Spain\' ORDER BY name')
    assert r.ok and r.rows == [("Ana",), ("Chloe",)] and r.columns == ["name"]


def test_writes_are_blocked(db):
    for sql in ["DELETE FROM customers", "DROP TABLE customers", "UPDATE customers SET name = 'x'",
                "INSERT INTO customers VALUES (9, 'x', 'y')", "PRAGMA writable_schema = 1"]:
        assert not db.run(sql).ok, sql
    assert db.run("SELECT COUNT(*) FROM customers").rows == [(3,)]


def test_write_hidden_inside_with_is_blocked_by_read_only_file(db):
    # Passes the SELECT/WITH check, but the read-only connection still refuses it.
    r = db.run("WITH x AS (SELECT 1) DELETE FROM customers")
    assert not r.ok
    assert db.run("SELECT COUNT(*) FROM customers").rows == [(3,)]


def test_only_one_statement(db):
    r = db.run("SELECT 1; DROP TABLE customers")
    assert not r.ok
    assert db.run("SELECT COUNT(*) FROM customers").rows == [(3,)]


def test_errors_are_returned_not_raised(db):
    r = db.run("SELECT nope FROM customers")
    assert r.error and "nope" in r.error


def test_timeout(db):
    slow = ("WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) "
            "SELECT COUNT(*) FROM n")  # never ends on its own
    r = db.run(slow, timeout_s=0.3)
    assert r.error and "timed out" in r.error
    assert db.run("SELECT 1").ok  # connection still usable afterwards


def test_row_limit_and_truncation(db):
    r = db.run("SELECT * FROM customers", max_rows=2)
    assert len(r.rows) == 2 and r.truncated


def test_is_empty_counts_all_null_rows(db):
    assert db.run("SELECT MAX(amount) FROM \"order\" WHERE amount > 1000").is_empty
    assert db.run("SELECT * FROM customers WHERE id = 99").is_empty
    assert not db.run("SELECT 1").is_empty


def test_strip_sql():
    assert strip_sql("-- comment\n  SELECT 1;  ") == "SELECT 1"
    assert strip_sql("/* hi */ SELECT 2;") == "SELECT 2"


def test_quote():
    assert quote("amount") == "amount"
    assert quote("Home Country") == '"Home Country"'
    assert quote("order") == '"order"'


def test_schema_text(db):
    text = describe(db, descriptions={"customers": {"home country": "where the customer lives"}})
    assert 'CREATE TABLE "order" (  -- 3 rows' in text
    assert "id INTEGER PRIMARY KEY" in text
    assert '"Home Country" TEXT  -- where the customer lives | e.g. \'Spain\', \'France\'' in text
    assert "FOREIGN KEY (customer_id) REFERENCES customers(id)" in text


def test_path_with_spaces(tmp_path, db_path):
    folder = tmp_path / "my data folder"
    folder.mkdir()
    target = folder / "shop copy.sqlite"
    target.write_bytes(db_path.read_bytes())
    with Database(target) as d:
        assert d.run("SELECT COUNT(*) FROM customers").rows == [(3,)]
