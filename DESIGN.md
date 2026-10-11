# Design notes

## The loop

1. `schema.describe()` reads the database once: tables, columns, types, primary and
   foreign keys, row counts, and three real example values per column. If the database
   ships BIRD-style `database_description/*.csv` files, their column descriptions go in
   as comments. The output is a set of commented `CREATE TABLE` statements, a format
   models have seen a lot of.
2. The model gets the schema, the question and the optional hint. It replies with
   either `EXPLORE` plus a query (look at the data, see up to 20 rows, continue) or
   `ANSWER` plus a query.
3. The answer query runs. If it raised an error, returned no rows, or returned a single
   row of only 0/NULL (the usual sign of a filter that matched nothing), the model is
   shown exactly that and asked to fix it. It may explore again first. If it resubmits
   the same query, the agent accepts that an empty result was intended.
4. Limits: 3 explorations, 2 fixes, 8 model calls per question.

The first `ANSWER` is stored separately from the final one. With temperature 0 that
first answer is exactly what the agent returns when self-correction is switched off, so
one benchmark run gives both numbers. `--no-fix` exists to check this directly.

## Safety

The model writes SQL that runs against someone's data, so `db.py` is the boundary:

- The file is opened with SQLite's `mode=ro`. Writes fail inside SQLite, whatever the
  query looks like (tested with `WITH x AS (SELECT 1) DELETE ...`).
- Queries must start with `SELECT` or `WITH`, and `sqlite3` refuses to run more than
  one statement per call.
- A progress handler cancels any query that runs longer than 30 s.
- At most 1,000 rows come back to the app; exploration shows the model 20.

## Evaluation

BIRD's metric is execution accuracy: run the predicted and gold SQL on the same
database and compare `set(rows)`. Row order, duplicates and column names do not count;
extra or missing columns do. `bench/run.py` implements the same rule and scores with
full results (no row limit) and a 60 s timeout per query.

- Test set: BIRD Mini-Dev, SQLite version (500 questions, 11 databases, 30% simple,
  50% moderate, 20% challenging). All 500 gold queries were checked to run on the
  downloaded databases (`python -m sqlagent gold-check`).
- Tuning set: the BIRD dev questions that are not in Mini-Dev (1,037). Prompt changes
  were checked on this set so the test questions are not used for tuning.
- BIRD's hint ("evidence") is passed to the model, as in the standard BIRD setting.
- The report gives 95% bootstrap confidence intervals, and counts how many answers
  self-correction fixed (wrong to right) and broke (right to wrong).

## Decisions

**A text protocol instead of native tool calling.** `EXPLORE`/`ANSWER` plus a SQL code
block works the same on Groq, Gemini and a 7B local model. Small local models are much
less reliable with JSON tool calls than with a code block.

**Compact history.** After each turn only the action and the query are kept, not the
model's explanation. Groq's free tier allows 8,000 tokens per minute and the biggest
schema is about 4,700 tokens, so every token in the history matters.

**Exploration is mostly up front.** Example values in the schema answer most "how is
this stored" questions without a model call. Exploration queries are there for the
rest, such as checking the exact spelling of a value that is not among the examples.

**Ollama goes through its native API.** The OpenAI-compatible endpoint cannot change
the context window, and the default is smaller than the larger BIRD schemas. Ollama
truncates the prompt without an error, which looks like the model ignoring half the
tables.

**Execution feedback only.** Self-correction reacts to things the database can prove:
errors and empty results. A second "does this answer look right?" model call would
catch more, but it doubles the cost per question and can talk the model out of correct
answers. I did not add it.

## Known limitations

- BIRD's gold SQL has errors. In a 20-question sample, 2 gold answers were wrong
  (one filters on a price that does not match the question). Every system is scored
  against the same labels, but the ceiling is below 100%.
- Execution feedback cannot catch a query that runs, returns rows, and is still wrong
  (wrong join, extra column). Those are most of the remaining errors.
- The schema is sent whole. A database with hundreds of tables would need a retrieval
  step that picks the relevant tables first.
