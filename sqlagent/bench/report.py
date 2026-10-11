"""Summarize benchmark runs into the numbers for the README.

For each run file it reports:
  * execution accuracy (EX) of the first answer   = agent WITHOUT self-correction
  * EX of the final answer                         = agent WITH self-correction
  * the difference, with a 95% confidence interval (bootstrap over questions)
  * how many questions self-correction fixed (wrong -> right) and broke (right -> wrong)
  * accuracy by difficulty, average LLM calls, tokens and time per question
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .bird import DIFFICULTIES


def load(path: Path) -> list[dict]:
    recs = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if not r.get("llm_error"):
                recs[r["question_id"]] = r  # last record wins if a question was re-run
    return list(recs.values())


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def bootstrap_ci(values: list[float], iters: int = 5000, seed: int = 0) -> tuple[float, float]:
    """95% confidence interval for the mean, by resampling questions with replacement."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


def summarize(recs: list[dict]) -> dict:
    n = len(recs)
    first = [r["correct_first"] for r in recs]
    final = [r["correct_final"] for r in recs]
    delta = [b - a for a, b in zip(first, final)]
    s = {
        "n": n,
        "model": recs[0]["model"] if recs else "",
        "ex_first": sum(first) / n,
        "ex_final": sum(final) / n,
        "ex_final_ci": bootstrap_ci([float(x) for x in final]),
        "delta": sum(delta) / n,
        "delta_ci": bootstrap_ci([float(d) for d in delta]),
        "fixed": sum(d == 1 for d in delta),
        "broke": sum(d == -1 for d in delta),
        "fix_attempted": sum(r["fixes"] > 0 for r in recs),
        "first_errors": sum(bool(r["first_error"]) for r in recs),
        "final_errors": sum(bool(r["final_error"]) for r in recs),
        "explored": sum(r["explores"] > 0 for r in recs),
        "avg_calls": sum(r["llm_calls"] for r in recs) / n,
        "avg_tokens": sum(r["prompt_tokens"] + r["completion_tokens"] for r in recs) / n,
        "avg_seconds": sum(r["seconds"] for r in recs) / n,
        "by_difficulty": {},
    }
    for d in DIFFICULTIES:
        sub = [r for r in recs if r["difficulty"] == d]
        if sub:
            s["by_difficulty"][d] = (len(sub), sum(r["correct_first"] for r in sub) / len(sub),
                                     sum(r["correct_final"] for r in sub) / len(sub))
    return s


def render(paths: list[Path]) -> str:
    out = []
    for p in paths:
        recs = load(p)
        if not recs:
            out.append(f"## {p.name}\n\n(no finished questions)\n")
            continue
        s = summarize(recs)
        lo, hi = s["ex_final_ci"]
        dlo, dhi = s["delta_ci"]
        out.append(f"## {p.stem}  ({s['model']}, {s['n']} questions)\n")
        out.append("| | Execution accuracy |")
        out.append("|---|---|")
        out.append(f"| Without self-correction (first answer) | {_pct(s['ex_first'])} |")
        out.append(f"| **With self-correction (final answer)** | **{_pct(s['ex_final'])}** (95% CI {_pct(lo)} - {_pct(hi)}) |")
        out.append(f"| Gain from self-correction | **{100 * s['delta']:+.1f} points** (95% CI {100 * dlo:+.1f} to {100 * dhi:+.1f}) |")
        out.append("")
        out.append("| Difficulty | Questions | First answer | Final answer |")
        out.append("|---|---|---|---|")
        for d, (k, a, b) in s["by_difficulty"].items():
            out.append(f"| {d} | {k} | {_pct(a)} | {_pct(b)} |")
        out.append("")
        out.append(f"- Self-correction was triggered on {s['fix_attempted']} questions: "
                   f"it **fixed {s['fixed']}** (wrong -> right) and broke {s['broke']} (right -> wrong).")
        out.append(f"- Queries that crashed with a SQL error: {s['first_errors']} on the first answer, "
                   f"{s['final_errors']} after self-correction.")
        out.append(f"- The agent ran exploration queries on {s['explored']} questions.")
        out.append(f"- Cost per question: {s['avg_calls']:.2f} LLM calls, {s['avg_tokens']:,.0f} tokens, "
                   f"{s['avg_seconds']:.1f}s on average.")
        out.append("")
    return "\n".join(out)
