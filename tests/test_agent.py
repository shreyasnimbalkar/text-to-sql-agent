from sqlagent.agent import Agent, parse_reply
from tests.conftest import ScriptedLLM


def sql(action, query, chatter="Let me think."):
    return f"{chatter}\n{action}\n```sql\n{query}\n```"


def make(db, replies, **kw):
    llm = ScriptedLLM(replies)
    return Agent(llm, db, "<schema>", **kw), llm


def test_parse_reply():
    assert parse_reply(sql("EXPLORE", "SELECT 1")) == ("explore", "SELECT 1")
    assert parse_reply(sql("ANSWER", "SELECT 2;")) == ("answer", "SELECT 2")
    assert parse_reply("```sql\nSELECT 3\n```") == ("answer", "SELECT 3")  # no action word -> answer
    assert parse_reply("ANSWER: SELECT name FROM t") == ("answer", "SELECT name FROM t")  # no code block
    assert parse_reply("I don't know") == ("answer", None)
    # The model mentions EXPLORE in its reasoning but the last action word wins.
    assert parse_reply("I could EXPLORE but no need.\nANSWER\n```sql\nSELECT 4\n```") == ("answer", "SELECT 4")


def test_correct_first_try(db):
    agent, llm = make(db, [sql("ANSWER", "SELECT COUNT(*) FROM customers")])
    r = agent.ask("How many customers?")
    assert r.sql == r.first_sql and r.result.rows == [(3,)]
    assert r.fixes == 0 and r.llm_calls == 1


def test_fixes_an_error(db):
    agent, llm = make(db, [
        sql("ANSWER", "SELECT COUNT(*) FROM customer"),  # wrong table name
        sql("ANSWER", "SELECT COUNT(*) FROM customers"),
    ])
    r = agent.ask("How many customers?")
    assert r.first_result.error and "no such table" in r.first_result.error
    assert r.result.rows == [(3,)] and r.fixes == 1
    # The model was shown the SQLite error message.
    assert "no such table: customer" in llm.sent[1][-1]["content"]


def test_count_of_zero_is_treated_as_suspicious(db):
    agent, llm = make(db, [
        sql("ANSWER", "SELECT COUNT(*) FROM customers WHERE \"Home Country\" = 'spain'"),
        sql("ANSWER", "SELECT COUNT(*) FROM customers WHERE \"Home Country\" = 'Spain'"),
    ])
    r = agent.ask("How many customers live in Spain?")
    assert r.first_result.rows == [(0,)] and r.fixes == 1
    assert r.result.rows == [(2,)]


def test_fixes_an_empty_result_after_exploring(db):
    agent, llm = make(db, [
        sql("ANSWER", "SELECT name FROM customers WHERE \"Home Country\" = 'spain'"),
        sql("EXPLORE", "SELECT DISTINCT \"Home Country\" FROM customers"),
        sql("ANSWER", "SELECT name FROM customers WHERE \"Home Country\" = 'Spain'"),
    ])
    r = agent.ask("Which customers live in Spain?")
    assert r.first_result.rows == [] and r.fixes == 1
    assert r.explores == 1
    assert set(r.result.rows) == {("Ana",), ("Chloe",)}
    assert "Spain" in llm.sent[2][-1]["content"]  # exploration result was shown to the model


def test_accepts_resubmitted_empty_answer(db):
    q = "SELECT name FROM customers WHERE id = 99"
    agent, llm = make(db, [sql("ANSWER", q), sql("ANSWER", q)])
    r = agent.ask("Who is customer 99?")
    assert r.sql == q and r.llm_calls == 2 and r.fixes == 1


def test_fix_budget(db):
    bad = sql("ANSWER", "SELECT nope FROM customers")
    agent, llm = make(db, [bad, bad, bad, bad], max_fixes=2)
    r = agent.ask("?")
    assert r.fixes == 2 and r.llm_calls == 3 and not r.result.ok


def test_self_correction_off(db):
    agent, llm = make(db, [sql("ANSWER", "SELECT nope FROM customers")], self_correct=False)
    r = agent.ask("?")
    assert r.llm_calls == 1 and r.fixes == 0 and not r.result.ok


def test_explore_budget(db):
    explore = sql("EXPLORE", "SELECT 1")
    agent, llm = make(db, [explore, explore, sql("ANSWER", "SELECT 2")], max_explore=1)
    r = agent.ask("?")
    assert r.explores == 2  # the second was refused...
    assert "no explorations left" in llm.sent[2][-1]["content"]  # ...and the model was told so
    assert r.steps[1].result is None
    assert r.result.rows == [(2,)]


def test_unreadable_reply_then_answer(db):
    agent, llm = make(db, ["Sorry, I am not sure.", sql("ANSWER", "SELECT 5")])
    r = agent.ask("?")
    assert r.steps[0].kind == "invalid" and r.result.rows == [(5,)]


def test_only_explored_uses_last_query(db):
    explore = sql("EXPLORE", "SELECT COUNT(*) FROM customers")
    agent, llm = make(db, [explore] * 3, max_explore=5, max_steps=3)
    r = agent.ask("How many customers?")
    assert r.sql == "SELECT COUNT(*) FROM customers" and r.result.rows == [(3,)]


def test_history_is_compact(db):
    long_chatter = "blah " * 500
    agent, llm = make(db, [sql("ANSWER", "SELECT nope", long_chatter), sql("ANSWER", "SELECT 1")])
    agent.ask("?")
    assistant_turn = llm.sent[1][2]
    assert assistant_turn["role"] == "assistant" and "blah" not in assistant_turn["content"]
