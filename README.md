# Text-to-SQL data agent

[![tests](https://github.com/shreyasnimbalkar/text-to-sql-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/shreyasnimbalkar/text-to-sql-agent/actions/workflows/tests.yml)

Ask a database a question in plain English and get a table back. The agent reads the
schema and some real values, writes a SQLite query, runs it, and when the query fails
or comes back empty it reads the error and tries again.

The point is to let people who don't write SQL query their own data. I measured it on
[BIRD](https://bird-bench.github.io/), a text-to-SQL benchmark built on real, messy
databases (schools, Formula 1, banking, card games, ...).

## Results

The full run on BIRD Mini-Dev (500 questions, SQLite) with `gpt-oss-120b` on Groq is in
progress. The free tier allows 8,000 tokens a minute, so it takes a while.

Development numbers so far, on 30 held-out BIRD dev questions that are not part of the
test set (too few to quote as a result, the 95% interval is about ±15 points):

| | Execution accuracy |
|---|---|
| First answer (no self-correction) | 73.3% |
| Final answer (with self-correction) | 76.7% |

A 10-question check with a local `qwen2.5-coder:7b` went from 40% to 50% with
self-correction. Smaller models make more mistakes the database can catch.

## How it works

```
question ─► schema + example values ─► model ─┬─ EXPLORE query ─► run, show rows ─┐
                                              │   ◄──────────────────────────────┘
                                              └─ ANSWER query ─► run
                                                                   │
                         error, no rows, or COUNT = 0 ◄────────────┤
                         (sent back to the model to fix)           │
                                                                   ▼
                                                             result table
```

- The schema is rendered as commented `CREATE TABLE` statements with three real
  values per column, so the model can see that a flag is stored as `'Y'` or a date as
  `'201309'` before it writes anything.
- The model can run up to 3 exploration queries (for example, to check how a name is
  spelled) and gets up to 2 rounds of fixes.
- Queries run on a read-only connection with a 30 second timeout. Only `SELECT`/`WITH`
  is accepted.
- It works with Groq, Gemini, OpenRouter, a local Ollama model, or through my
  [Switchboard](https://github.com/shreyasnimbalkar/switchboard-llm-gateway) gateway.

[DESIGN.md](DESIGN.md) covers the loop, the evaluation setup and the trade-offs.

## Quick start

```powershell
git clone https://github.com/shreyasnimbalkar/text-to-sql-agent.git
cd text-to-sql-agent
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
copy .env.example .env                       # add a Groq or Gemini key, or use Ollama
.venv\Scripts\python scripts\download_bird.py  # BIRD databases, ~350 MB download
```

Ask a question:

```powershell
.venv\Scripts\python -m sqlagent ask --db data\bird\dev_databases\california_schools\california_schools.sqlite "How many charter schools are in Alameda county?"
```

It prints each step (explorations, the answer query, any fixes) and the final table.
Use `--model ollama:qwen2.5-coder:7b` to run fully offline.

Web app:

```powershell
.venv\Scripts\streamlit run app.py
```

Pick one of the BIRD databases or upload your own `.sqlite` file.

## Running the benchmark

```powershell
.venv\Scripts\python -m sqlagent bench --model groq:openai/gpt-oss-120b
.venv\Scripts\python -m sqlagent report runs\groq_openai_gpt-oss-120b.jsonl
```

Results are appended to a JSONL file one question at a time, so a run that hits a
daily rate limit can be restarted with the same command and picks up where it stopped.
`--limit N` evaluates a stratified sample, `--no-fix` turns self-correction off.

## Tests

```powershell
.venv\Scripts\python -m pytest -q
```

The tests run offline. The agent tests use a scripted fake model (first reply has a
typo in a table name, second reply fixes it) against a small temporary database, and
check that writes, multiple statements and runaway queries are all blocked.

## Layout

```
sqlagent/
  agent.py       the explore / answer / fix loop
  db.py          read-only SQLite access with timeouts
  schema.py      schema + example values as commented CREATE TABLEs
  llm.py         chat client for Groq, Gemini, OpenRouter, Ollama, Switchboard
  prompts.py     every prompt in one place
  cli.py         ask / bench / report / gold-check
  bench/         BIRD loading, scoring and the results report
app.py           Streamlit web app
scripts/         download_bird.py
tests/
```

## License

MIT. BIRD is CC BY-SA 4.0 and is downloaded separately, not included here.
