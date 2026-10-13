"""Command line interface.

  python -m sqlagent ask   --db path/to/file.sqlite "How many customers are in Spain?"
  python -m sqlagent bench --model groq:openai/gpt-oss-120b --limit 100
  python -m sqlagent report runs/*.jsonl
  python -m sqlagent gold-check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_MODEL = "groq:openai/gpt-oss-120b"


def _extra(args) -> dict:
    return {"reasoning_effort": args.reasoning} if args.reasoning else {}


def cmd_ask(args) -> int:
    from .agent import Agent
    from .bench.bird import read_description_folder
    from .db import Database
    from .llm import LLM
    from .schema import describe

    db = Database(args.db)
    llm = LLM(args.model, extra=_extra(args), timeout_s=args.timeout)
    # BIRD-style column descriptions, if the database comes with them.
    descriptions = read_description_folder(db.path.parent / "database_description")
    agent = Agent(llm, db, describe(db, descriptions), max_explore=args.max_explore, max_fixes=args.max_fixes,
                  self_correct=not args.no_fix)

    def show(step):
        label = {"explore": "Exploring", "answer": "Answer query", "invalid": "Unreadable reply"}[step.kind]
        print(f"\n--- {label} ---")
        if step.sql:
            print(step.sql)
        if step.result is not None:
            print("\n" + step.result.preview(max_rows=5))
        if step.kind == "answer" and step.feedback:
            print("\n(not good enough - asking the model to fix it)")

    res = agent.ask(args.question, args.hint or "", on_step=show)
    print("\n=== Final answer ===")
    print(res.sql or "(no query)")
    if res.result is not None:
        print("\n" + res.result.preview(max_rows=args.rows, max_chars=10_000))
    print(f"\n{res.llm_calls} LLM calls, {res.prompt_tokens + res.completion_tokens:,} tokens, {res.seconds:.1f}s")
    return 0


def cmd_bench(args) -> int:
    from .bench import bird, run

    questions = bird.load_questions(args.questions)
    if args.db_id:
        questions = [q for q in questions if q.db_id == args.db_id]
    if args.limit:
        questions = bird.sample(questions, args.limit, seed=args.seed)
    name = args.out or f"runs/{args.model.replace(':', '_').replace('/', '_')}{'_nofix' if args.no_fix else ''}.jsonl"
    run.run(args.model, Path(name), questions, self_correct=not args.no_fix, max_explore=args.max_explore,
            max_fixes=args.max_fixes, use_descriptions=not args.no_descriptions, extra=_extra(args),
            workers=args.workers, timeout_s=args.timeout)
    return 0


def cmd_report(args) -> int:
    from .bench import report

    text = report.render([Path(p) for p in args.files])
    print(text)
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(text, encoding="utf-8")
        print(f"saved to {args.save}")
    return 0


def cmd_gold_check(args) -> int:
    from .bench import bird, run

    run.gold_check(bird.load_questions(args.questions))
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv(Path.cwd() / ".env")
    # Windows terminals default to a legacy code page; database values can contain any character.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(prog="sqlagent", description="Text-to-SQL data agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    def agent_opts(sp):
        sp.add_argument("--model", default=DEFAULT_MODEL, help=f"provider:model (default {DEFAULT_MODEL})")
        sp.add_argument("--timeout", type=float, help="seconds to wait for each model reply (default 60, Ollama 300)")
        sp.add_argument("--reasoning", choices=["low", "medium", "high"], default="low",
                        help="reasoning effort for reasoning models like gpt-oss (default low)")
        sp.add_argument("--max-explore", type=int, default=3, help="max exploration queries per question")
        sp.add_argument("--max-fixes", type=int, default=2, help="max self-correction rounds per question")
        sp.add_argument("--no-fix", action="store_true", help="switch self-correction off")

    a = sub.add_parser("ask", help="ask one question about a SQLite database")
    a.add_argument("question")
    a.add_argument("--db", required=True, help="path to a .sqlite file")
    a.add_argument("--hint", help="optional extra knowledge, e.g. a definition")
    a.add_argument("--rows", type=int, default=20, help="rows of the result to print")
    agent_opts(a)
    a.set_defaults(fn=cmd_ask)

    b = sub.add_parser("bench", help="run the BIRD benchmark")
    b.add_argument("--questions", help="question file (default: data/bird/mini_dev_sqlite.json)")
    b.add_argument("--limit", type=int, help="evaluate a stratified sample of this many questions")
    b.add_argument("--seed", type=int, default=0, help="sample seed")
    b.add_argument("--db-id", help="only questions about this database")
    b.add_argument("--out", help="results file (.jsonl); default runs/<model>.jsonl")
    b.add_argument("--workers", type=int, default=1, help="questions to run at the same time (default 1)")
    b.add_argument("--no-descriptions", action="store_true", help="don't give the model BIRD's column descriptions")
    agent_opts(b)
    b.set_defaults(fn=cmd_bench)

    r = sub.add_parser("report", help="summarize benchmark result files")
    r.add_argument("files", nargs="+")
    r.add_argument("--save", help="also write the report to this Markdown file")
    r.set_defaults(fn=cmd_report)

    g = sub.add_parser("gold-check", help="check that BIRD's gold SQL runs on the local databases")
    g.add_argument("--questions")
    g.set_defaults(fn=cmd_gold_check)

    args = p.parse_args(argv)
    if getattr(args, "model", "").startswith("ollama:"):
        args.reasoning = None  # Ollama ignores it; keep the request clean
    return args.fn(args)
