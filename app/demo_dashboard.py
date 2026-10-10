"""Audience-friendly viewer for the six demo stages and saved artifacts."""

import json
from pathlib import Path

import streamlit as st
from learning.lesson_store import LessonStore


st.set_page_config(page_title="Agent Improvement Lab", layout="wide")
st.title("Agent Improvement Lab")
st.caption("Fictional orders · inspectable decisions · measured procedural updates")

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
    tabs = st.tabs(["Graph", "Failure trace", "Evaluation", "Improvement job", "Heldout", "Safety"])
    with tabs[0]:
        st.subheader("Bounded LangGraph agent")
        st.code("decide → dispatch read-only tool → decide\ndecide → final recommendation or clarification")
        st.write("Gemini chooses a read-only information tool or a recommendation. The dispatcher and final policy check enforce safety bounds.")
    with tabs[1]:
        st.subheader("Follow one execution")
        trace_dir = run_dir / "traces" / "baseline-train"
        traces = sorted(trace_dir.glob("*.json")) if trace_dir.exists() else []
        if traces:
            chosen = st.selectbox("Case trace", traces, format_func=lambda p: p.stem)
            trajectory = json.loads(chosen.read_text())
            st.write("Skill", trajectory.get("skill"), "Memory IDs", trajectory.get("retrieved_memory_ids"))
            if trajectory.get("langsmith_run_id"):
                st.caption(f"LangSmith run ID: {trajectory['langsmith_run_id']}")
            st.json(trajectory)
        else:
            st.info("Run `uv run python demo.py baseline` to save baseline traces.")
    with tabs[2]:
        st.subheader("Baseline evaluation")
        show_report("baseline-train.json")
    with tabs[3]:
        st.subheader("Evidence, skill diff, paired validation and gate")
        jobs = sorted((run_dir / "jobs").glob("*.json")) if (run_dir / "jobs").exists() else []
        if jobs:
            selected_job = st.selectbox("Improvement job", jobs, format_func=lambda p: json.loads(p.read_text()).get("case_id", p.stem))
            job = json.loads(selected_job.read_text())
            st.json({key: value for key, value in job.items() if key != "diff"})
            if job.get("diff"):
                st.code(job["diff"], language="diff")
        else:
            st.info("Run baseline, then diagnose/propose/validate/promote to create a job.")
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
