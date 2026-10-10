"""Audience-friendly viewer for the six demo stages and saved artifacts."""

import json
from pathlib import Path

import streamlit as st
from learning.lesson_store import LessonStore


st.set_page_config(page_title="Agent Improvement Lab", layout="wide")
st.title("Agent Improvement Lab")
st.caption("One agent · two versions · measured learning")

latest = Path("runs/latest.txt")
default = latest.read_text().strip() if latest.exists() else ""
selected = st.sidebar.text_input("Run directory", default)
run_dir = Path(selected) if selected else None


def read_json(name: str):
    if run_dir is None:
        return None
    path = run_dir / name
    return json.loads(path.read_text()) if path.exists() else None


def show_report(name: str):
    report = read_json(name)
    if report is None:
        st.info(f"Run the corresponding stage to create {name}.")
        return
    if "summary" in report:
        score = report["summary"]
        st.metric("Passed", f"{score['passed']}/{score['total']}")
        st.dataframe(report["grades"], use_container_width=True)
    else:
        st.json(report)


if run_dir is None:
    st.info("Run `uv run python demo.py demo` first, or enter an existing run directory in the sidebar.")
else:
    st.sidebar.caption(str(run_dir.resolve()))
    tabs = st.tabs(["1 Baseline", "2 Failure", "3 Evaluate", "4 Learn", "5 Verify", "6 Safeguard"])
    with tabs[0]:
        st.subheader("The baseline LangGraph agent")
        st.code("plan_tools → get_order_details → search_policy → create_resolution")
        st.write("Order and policy functions read fictional records. Gemini chooses policies and proposes a resolution.")
    with tabs[1]:
        st.subheader("Follow one execution")
        trace_dir = run_dir / "traces" / "baseline-train"
        traces = sorted(trace_dir.glob("*.json")) if trace_dir.exists() else []
        if traces:
            chosen = st.selectbox("Case trace", traces, format_func=lambda p: p.stem)
            st.json(json.loads(chosen.read_text()))
        else:
            st.info("Run `uv run python demo.py baseline` to save baseline traces.")
    with tabs[2]:
        st.subheader("Baseline evaluation")
        show_report("baseline-train.json")
    with tabs[3]:
        st.subheader("Candidate lesson and validation gate")
        show_report("candidate.json")
        if (run_dir / "lessons.sqlite").exists():
            st.dataframe(LessonStore(run_dir / "lessons.sqlite").list_versions(), use_container_width=True)
        with st.expander("Validation case scores"):
            show_report("candidate-validation.json")
    with tabs[4]:
        st.subheader("Untouched holdout comparison")
        show_report("comparison.json")
    with tabs[5]:
        st.subheader("Bad update and rollback control")
        show_report("bad-update.json")
        st.caption("Rejected lessons never become the active version. `uv run python demo.py rollback` restores the prior version if an active lesson must be withdrawn.")
