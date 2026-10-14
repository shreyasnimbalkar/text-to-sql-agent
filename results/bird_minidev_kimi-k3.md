## Kimi K3 on BIRD Mini-Dev (all 500 questions)

| | Execution accuracy |
|---|---|
| Without self-correction (first answer) | 66.6% |
| **With self-correction (final answer)** | **68.2%** (95% CI 64.0% - 72.4%) |
| Gain from self-correction | **+1.6 points** (95% CI +0.6 to +2.8) |

| Difficulty | Questions | First answer | Final answer |
|---|---|---|---|
| simple | 148 | 79.1% | 79.7% |
| moderate | 250 | 64.0% | 66.4% |
| challenging | 102 | 54.9% | 55.9% |

- Self-correction was triggered on 17 questions: it **fixed 8** (wrong -> right) and broke 0 (right -> wrong).
- Queries that crashed with a SQL error: 2 on the first answer, 0 after self-correction.
- The agent ran exploration queries on 27 questions.
- Cost per question: 1.12 LLM calls, 4,058 tokens, 80.8s on average.

Why the 159 wrong answers were wrong:

| Reason | Questions |
|---|---|
| wrong values (filter, join or calculation) | 104 |
| wrong columns (extra, missing or merged) | 34 |
| too many or too few rows (DISTINCT, LIMIT, ties) | 15 |
| same numbers, different rounding | 4 |
| gold answer is empty or fails (benchmark label problem) | 1 |
| query returns no data | 1 |
