# Text-to-SQL data agent

[![tests](https://github.com/shreyasnimbalkar/text-to-sql-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/shreyasnimbalkar/text-to-sql-agent/actions/workflows/tests.yml)

Ask a database a question in plain English and get a table back. The agent reads the
schema and some real values, writes a SQLite query, runs it, and when the query fails
or comes back empty it reads the error and tries again.

The point is to let people who don't write SQL query their own data. I measured it on
[BIRD](https://bird-bench.github.io/), a text-to-SQL benchmark built on real, messy
databases (schools, Formula 1, banking, card games, ...).

## Results

BIRD Mini-Dev, SQLite version: all 500 questions over 11 databases, scored with BIRD's
execution accuracy (the query must return the same rows as the reference query). The
model is Kimi K3 through NVIDIA's free API, temperature 0, with BIRD's hints.

| | Execution accuracy |
|---|---|
| First answer (self-correction off) | 66.6% |
| Final answer (self-correction on) | **68.2%** (95% CI 64.0 to 72.4) |
| Gain from self-correction | **+1.6 points** (95% CI +0.6 to +2.8) |

| Difficulty | Questions | First answer | Final answer |
|---|---|---|---|
| simple | 148 | 79.1% | 79.7% |
| moderate | 250 | 64.0% | 66.4% |
| challenging | 102 | 54.9% | 55.9% |

Self-correction kicked in on 17 questions. It turned 8 wrong answers into right ones and
never turned a right answer wrong. Queries that crashed went from 2 to 0. The first
answer and the final answer come from the same run, so the comparison is paired.

A second model, GLM-5.3, got 68.9% with a +3.5 point gain (16 fixed, 0 broken), but only
on 454 of the 500 questions because NVIDIA throttled it before the run finished. Details
in [results/](results/).

What the remaining mistakes look like (159 wrong answers):

| Reason | Questions |
|---|---|
| Wrong values: a filter, join or calculation is off | 104 |
| Wrong columns: extra, missing, or two glued together | 34 |
| Too many or too few rows (DISTINCT, LIMIT, ties) | 15 |
| Same numbers, different rounding | 4 |
| Other (empty gold answer, empty prediction) | 2 |

Almost all of them are queries that run fine and return plausible data, which is the
kind of mistake execution feedback can't see. That is also why the gain from
self-correction is small with a strong model: it rarely writes a query that fails.

Prompts were tuned on 1,037 BIRD dev questions that are not in Mini-Dev, never on the
test questions. An average question took 1.12 model calls and about 4,000 tokens.

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
- It works with NVIDIA's free API, Groq, Gemini, OpenRouter, a local Ollama model, or through my
  [Switchboard](https://github.com/shreyasnimbalkar/switchboard-llm-gateway) gateway.

[DESIGN.md](DESIGN.md) covers the loop, the evaluation setup and the trade-offs.

## Quick start

```powershell
git clone https://github.com/shreyasnimbalkar/text-to-sql-agent.git
cd text-to-sql-agent
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
copy .env.example .env                       # add an NVIDIA, Groq or Gemini key, or use Ollama
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
.venv\Scripts\python -m sqlagent bench --model nvidia:moonshotai/kimi-k3 --workers 4 --timeout 180
.venv\Scripts\python -m sqlagent report runs
vidia_moonshotai_kimi-k3.jsonl --why
```

Results are appended to a JSONL file one question at a time, so a run that hits a
daily rate limit can be restarted with the same command and picks up where it stopped.
`--limit N` evaluates a stratified sample, `--no-fix` turns self-correction off, and
`--why` adds the breakdown of wrong answers.

Free tiers made this the slowest part of the project. Groq allows 200,000 tokens a day
for gpt-oss-120b, about 50 questions. NVIDIA has no daily cap but throttles a model for
a long time if you go over about 40 requests a minute, and its response times swung
between 5 and 120 seconds. Hence the parallel workers, the per-request timeout, and a
runner that skips a question after repeated 429s or timeouts and retries it on the next
run instead of stopping.

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
  llm.py         chat client for NVIDIA, Groq, Gemini, OpenRouter, Ollama, Switchboard
  prompts.py     every prompt in one place
  cli.py         ask / bench / report / gold-check
  bench/         BIRD loading, scoring and the results report
app.py           Streamlit web app
scripts/         download_bird.py
results/         benchmark reports
tests/
```

## License

MIT. BIRD is CC BY-SA 4.0 and is downloaded separately, not included here.
