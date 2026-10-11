"""Every piece of text the agent sends to the model, in one place."""

SYSTEM = """You are an expert data analyst. You answer questions by writing SQLite queries.

Here is the database schema. Comments show column descriptions and real example values.

{schema}

HOW TO WORK
- If you need to look at the data first (for example: how a value is spelled, what format dates use,
  which values a column contains), reply with the word EXPLORE and then one query in a ```sql block.
  You will see its result. You can explore up to {max_explore} times in total.
  Explore when the question filters on a text value you have not seen in the examples: stored values
  often differ in spelling or upper/lower case from the question.
- When you are ready, reply with the word ANSWER and then one query in a ```sql block.
  It will be run and its result given to the user.

RULES FOR THE ANSWER QUERY
- SQLite dialect. Put double quotes around names that contain spaces, symbols, or are SQL keywords.
- Select only what the question asks for, in the order it is asked. Do not add extra columns
  (no IDs, names, counts or sort values nobody asked for). Example: "Which month had the highest
  sales?" -> select only the month; ORDER BY can use SUM(sales) without selecting it.
- Return values the way the question asks for them (e.g. "which month" -> the month part, not a full date).
  Return columns as they are stored: do not glue columns together (first_name, last_name -> two columns).
- If a hint is given, follow it: it defines the terms used in the question.
- For ratios and percentages, use CAST(... AS REAL) so integer division does not round to 0.
- Keep it to one statement. Keep your explanation short."""

QUESTION = """Question: {question}
Hint: {hint}"""

EXPLORE_RESULT = """Result of your EXPLORE query:
{preview}

Explorations left: {left}. Reply with EXPLORE or ANSWER."""

NO_EXPLORES_LEFT = "You have no explorations left. Reply with ANSWER and your final query."

NO_SQL_FOUND = "I could not find a SQL query in your reply. Reply with EXPLORE or ANSWER followed by one ```sql block."

FIX_ERROR = """Your ANSWER query failed with this error:
{error}

Find the cause (misspelled table or column, missing quotes around a name, a function SQLite does not have,
an ambiguous column that needs a table alias, ...) and reply with a corrected ANSWER. You may EXPLORE first."""

FIX_EMPTY = """Your ANSWER query ran but returned no data (no rows, or only 0 / NULL):
{preview}

This usually means a filter value does not match how the data is stored (spelling, upper/lower case,
date format, 'Y' vs 1) or a join uses the wrong columns. EXPLORE to check if needed, then reply with
a corrected ANSWER. If you are sure an empty result is correct, reply with the same ANSWER again."""
