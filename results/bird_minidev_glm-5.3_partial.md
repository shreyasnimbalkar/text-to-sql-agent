## GLM-5.3 on 454 of the 500 Mini-Dev questions (partial run)

The other 46 (all 32 `financial` questions and 14 `california_schools` ones) were not run: NVIDIA throttled this model for more than a day after the run went over its rate limit. The missing questions are harder than average, so this accuracy is probably a little optimistic. The self-correction comparison is paired on the same 454 questions, so it is unaffected.

| | Execution accuracy |
|---|---|
| Without self-correction (first answer) | 65.4% |
| **With self-correction (final answer)** | **68.9%** (95% CI 64.8% - 73.3%) |
| Gain from self-correction | **+3.5 points** (95% CI +2.0 to +5.3) |

| Difficulty | Questions | First answer | Final answer |
|---|---|---|---|
| simple | 141 | 78.0% | 81.6% |
| moderate | 221 | 61.5% | 65.6% |
| challenging | 92 | 55.4% | 57.6% |

- Self-correction was triggered on 28 questions: it **fixed 16** (wrong -> right) and broke 0 (right -> wrong).
- Queries that crashed with a SQL error: 8 on the first answer, 0 after self-correction.
- The agent ran exploration queries on 22 questions.
- Cost per question: 1.15 LLM calls, 4,135 tokens, 13.8s on average.

Why the 141 wrong answers were wrong:

| Reason | Questions |
|---|---|
| wrong values (filter, join or calculation) | 86 |
| wrong columns (extra, missing or merged) | 32 |
| too many or too few rows (DISTINCT, LIMIT, ties) | 14 |
| same numbers, different rounding | 7 |
| gold answer is empty or fails (benchmark label problem) | 1 |
| query returns no data | 1 |
