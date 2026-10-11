"""Web demo: ask a database a question in plain English.

Run:  .venv\\Scripts\\streamlit run app.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from sqlagent.agent import Agent
from sqlagent.bench.bird import DATA_DIR, read_description_folder
from sqlagent.db import Database
from sqlagent.llm import LLM, LLMError
from sqlagent.schema import describe

load_dotenv(Path(__file__).parent / ".env")
st.set_page_config(page_title="Ask your database", layout="wide")


@st.cache_data(show_spinner="Reading the database structure...")
def schema_text(path: str) -> str:
    with Database(path) as db:
        return describe(db, read_description_folder(Path(path).parent / "database_description"))


def bundled_databases() -> dict[str, str]:
    root = DATA_DIR / "dev_databases"
    if not root.exists():
        return {}
    return {p.stem: str(p) for p in sorted(root.glob("*/*.sqlite"))}


with st.sidebar:
    st.header("Database")
    dbs = bundled_databases()
    uploaded = st.file_uploader("Upload a SQLite file", type=["sqlite", "db", "sqlite3"])
    if uploaded is not None:
        tmp = Path(tempfile.gettempdir()) / f"sqlagent_{uploaded.name}"
        tmp.write_bytes(uploaded.getvalue())
        db_file = str(tmp)
    elif dbs:
        db_file = dbs[st.selectbox("Or pick a BIRD example database", list(dbs))]
    else:
        st.info("Upload a .sqlite file, or run scripts/download_bird.py to get the example databases.")
        st.stop()

    st.header("Model")
    model = st.text_input("provider:model", os.environ.get("SQLAGENT_MODEL", "groq:openai/gpt-oss-120b"))
    self_correct = st.checkbox("Self-correction", value=True, help="Let the agent fix queries that fail or return nothing.")

    with st.expander("Database structure"):
        st.code(schema_text(db_file), language="sql")

st.title("Ask your database")
st.caption("Type a question in plain English. The agent looks at the data, writes SQL, runs it, and fixes its own mistakes.")

question = st.text_input("Question", placeholder="e.g. How many schools in Alameda county are charter schools?")
hint = st.text_input("Extra knowledge (optional)", placeholder="e.g. 'charter school' means \"Charter School (Y/N)\" = 1")

if st.button("Ask", type="primary", disabled=not question):
    db = Database(db_file)
    try:
        extra = {} if model.startswith("ollama:") else {"reasoning_effort": "low"}
        agent = Agent(LLM(model, extra=extra), db, schema_text(db_file), self_correct=self_correct)
        steps_box = st.status("Working...", expanded=True)

        def show(step):
            with steps_box:
                if step.kind == "explore":
                    st.markdown("**Looking at the data**")
                elif step.kind == "answer":
                    st.markdown("**Answer query**")
                else:
                    st.markdown("**Could not read the model's reply, asking again**")
                if step.sql:
                    st.code(step.sql, language="sql")
                if step.result is not None:
                    if step.result.ok:
                        st.caption(f"{len(step.result.rows)} rows")
                    else:
                        st.error(step.result.error)
                if step.kind == "answer" and step.feedback:
                    st.warning("That doesn't look right. Asking the agent to fix it.")

        res = agent.ask(question, hint, on_step=show)
        steps_box.update(label=f"Done: {res.llm_calls} model calls, {res.seconds:.1f}s", state="complete", expanded=False)

        if res.result is None:
            st.error("The agent did not produce a query.")
        elif not res.result.ok:
            st.error(f"The query still fails: {res.result.error}")
        else:
            st.subheader("Answer")
            st.dataframe(pd.DataFrame(res.result.rows, columns=res.result.columns), use_container_width=True)
            if res.result.truncated:
                st.caption("Showing the first 1,000 rows.")
            if res.fixes:
                st.info(f"The agent corrected itself {res.fixes} time(s) before getting this answer.")
        with st.expander("SQL used"):
            st.code(res.sql or "", language="sql")
    except LLMError as e:
        st.error(f"Could not reach the model: {e}")
    finally:
        db.close()
