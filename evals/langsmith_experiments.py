"""Publish held-out datasets and real Gemini experiments to LangSmith."""

from __future__ import annotations

import os
from pathlib import Path

from langsmith import Client

from .evaluators import load_cases


def compare_in_langsmith(agent, run_dir: Path, candidate_rules: list[str]) -> dict:
    if not os.getenv("LANGSMITH_API_KEY"):
        raise RuntimeError("Set LANGSMITH_API_KEY to publish LangSmith experiments")
    client = Client()
    cases = load_cases("heldout")
    case_by_id = {case["id"]: case for case in cases}
    dataset_name = f"support-agent-heldout-{run_dir.name}"
    if not client.has_dataset(dataset_name=dataset_name):
        client.create_dataset(dataset_name, description="Fictional holdout cases for the agent improvement demo")
        client.create_examples(dataset_name=dataset_name, examples=[
            {
                "inputs": {"case_id": case["id"], "order_id": case["order_id"], "message": case["message"]},
                "outputs": {"expected_action": case["expected_action"], "required_policy": case["required_policy"]},
                "metadata": {"split": "heldout"},
            }
            for case in cases
        ])

    def target_for(rules: list[str], label: str):
        def target(inputs: dict) -> dict:
            case = case_by_id[inputs["case_id"]]
            trace = agent.run_case(case, rules, version=label)
            result = trace["steps"][-1]["output"]
            return {
                "action": result["action"],
                "policy_ids": result["policy_ids"],
                "customer_reply": result["customer_reply"],
                "tool_names": [step.get("name") for step in trace["steps"] if step.get("type") == "tool"],
                "retrieved_policy_ids": [policy["id"] for step in trace["steps"]
                                         if step.get("name") == "search_policy" for policy in step["output"]],
            }
        return target

    def resolution_correct(run, example):
        return {"key": "resolution_correct", "score": int(
            (run.outputs or {}).get("action") == (example.outputs or {}).get("expected_action")
        )}

    def required_tool_usage(run, example):
        return {"key": "required_tool_usage", "score": int(
            {"get_order_details", "search_policy", "create_resolution"} <=
            set((run.outputs or {}).get("tool_names", []))
        )}

    def policy_compliance(run, example):
        output = run.outputs or {}
        required = (example.outputs or {}).get("required_policy")
        cited = set(output.get("policy_ids", []))
        retrieved = set(output.get("retrieved_policy_ids", []))
        return {"key": "policy_compliance", "score": int(required in cited and cited <= retrieved)}

    evaluators = [resolution_correct, required_tool_usage, policy_compliance]
    experiments = {}
    for label, rules in (("baseline", []), ("learned", candidate_rules)):
        results = client.evaluate(
            target_for(rules, label),
            data=dataset_name,
            evaluators=evaluators,
            experiment_prefix=f"support-agent-{label}",
            description=f"{label} Gemini support agent on untouched holdout cases",
            max_concurrency=1,
        )
        experiments[label] = getattr(results, "experiment_name", None) or f"support-agent-{label}"
    return {"dataset": dataset_name, "experiments": experiments}
