"""Publish already-run heldout outputs to LangSmith without reusing heldout for tuning."""
from __future__ import annotations

import json
import os
from pathlib import Path

from langsmith import Client

from .evaluators import load_cases


def publish_saved_comparison(run_dir: Path) -> dict:
    if not os.getenv("LANGSMITH_API_KEY"):
        raise RuntimeError("Set LANGSMITH_API_KEY to publish saved experiments")
    run_dir = Path(run_dir)
    if not (run_dir / "comparison.json").exists():
        raise ValueError("Run the heldout comparison once before publishing")
    client = Client()
    cases = load_cases("heldout")
    dataset_name = f"support-agent-heldout-{run_dir.name}"
    if not client.has_dataset(dataset_name=dataset_name):
        client.create_dataset(dataset_name, description="Fictional heldout support cases")
        client.create_examples(dataset_name=dataset_name, examples=[{
            "inputs": {"case_id": case["id"]},
            "outputs": {"expected_action": case["expected_action"], "required_policy": case["required_policy"]},
            "metadata": {"split": "heldout", "fictional": True},
        } for case in cases])

    def target(label: str):
        def read_saved(inputs: dict) -> dict:
            path = run_dir / "traces" / label / f"{inputs['case_id']}.json"
            trajectory = json.loads(path.read_text())
            return {"action": trajectory["resolution"]["action"],
                    "policy_ids": trajectory["resolution"].get("policy_ids", []),
                    "trace_id": trajectory["trace_id"],
                    "langsmith_run_id": trajectory.get("langsmith_run_id")}
        return read_saved

    def action_correct(run, example):
        return {"key": "action_correct", "score": int((run.outputs or {}).get("action") ==
                (example.outputs or {}).get("expected_action"))}

    def policy_cited(run, example):
        return {"key": "policy_cited", "score": int((example.outputs or {}).get("required_policy") in
                (run.outputs or {}).get("policy_ids", []))}

    experiments = {}
    for label in ("baseline-heldout", "learned-heldout"):
        results = client.evaluate(target(label), data=dataset_name,
                                  evaluators=[action_correct, policy_cited],
                                  experiment_prefix=f"support-agent-{label}", max_concurrency=1,
                                  description="Evaluation of saved Gemini outputs; no new model calls")
        experiments[label] = getattr(results, "experiment_name", None) or label
    return {"dataset": dataset_name, "experiments": experiments,
            "note": "Published saved outputs; heldout was not rerun"}
