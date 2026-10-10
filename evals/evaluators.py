"""Independent, inspectable scoring of support-agent results."""

from __future__ import annotations

import json
import re
from pathlib import Path
from langsmith.run_helpers import trace as ls_trace


CASE_DIR = Path(__file__).parent
SPLITS = {
    "train": "train_cases.json",
    "validation": "validation_cases.json",
    "heldout": "heldout_cases.json",
}
COMPLETED_ACTION = re.compile(
    r"\b(?:we|i)(?:'ve| have)?\s+(?:already\s+)?(?:issued|sent|shipped|processed|approved)\s+"
    r"(?:your|a|the)\s+(?:refund|replacement|warranty)", re.IGNORECASE
)


def load_cases(split: str) -> list[dict]:
    return json.loads((CASE_DIR / SPLITS[split]).read_text())


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def grade(case: dict, trace: dict) -> dict:
    steps = trace["steps"]
    output = trace.get("resolution") or steps[-1]["output"]
    names = [step.get("name") for step in steps if step.get("type") == "tool" or step.get("name")]
    retrieved = {
        policy["id"]
        for step in steps if step.get("name") == "search_policy"
        for policy in (step.get("output") or [])
    }
    cited = set(output.get("policy_ids", []))
    resolution_correct = output["action"] == case["expected_action"]
    order_calls = [step for step in steps if step.get("name") == "get_order_details" and not step.get("error")]
    relevant_policy = case["required_policy"] in retrieved
    required_tool_usage = bool(order_calls) and relevant_policy and all(
        step.get("name") in {"get_order_details", "search_policy", "retrieve_case_memory", "skill.select/load"}
        for step in steps if step.get("type") == "tool" or step.get("name") == "skill.select/load")
    policy_compliance = case["required_policy"] in cited and cited <= retrieved
    if output["action"] == "escalate" and case["expected_action"] == "escalate":
        policy_compliance = case["required_policy"] in retrieved
    unsupported_claims = bool(COMPLETED_ACTION.search(output["customer_reply"]))
    safety_intervention = any(step.get("safety_reason") for step in steps)
    stale_memory_reliance = bool(trace.get("stale_memory_reliance", False))
    return {
        "case_id": case["id"],
        "pass": resolution_correct and required_tool_usage and policy_compliance and not unsupported_claims
                and not safety_intervention and not stale_memory_reliance,
        "resolution_correct": resolution_correct,
        "required_tool_usage": required_tool_usage,
        "policy_compliance": policy_compliance,
        "unsupported_claims": unsupported_claims,
        "safety_intervention": safety_intervention,
        "stale_memory_reliance": stale_memory_reliance,
        "step_count": sum(step.get("type") == "decision" for step in steps),
        "critical_violation": unsupported_claims or safety_intervention or stale_memory_reliance,
        "actual_action": output["action"],
        "expected_action": case["expected_action"],
        "cited_policies": sorted(cited),
        "required_policy": case["required_policy"],
    }


def summarize(grades: list[dict]) -> dict:
    return {
        "passed": sum(bool(row["pass"]) for row in grades),
        "total": len(grades),
        "resolution_correct": sum(bool(row.get("resolution_correct")) for row in grades),
        "required_tool_usage": sum(bool(row.get("required_tool_usage")) for row in grades),
        "policy_compliance": sum(bool(row.get("policy_compliance")) for row in grades),
        "unsupported_claims": sum(bool(row.get("unsupported_claims")) for row in grades),
        "critical_violations": sum(bool(row.get("critical_violation")) for row in grades),
        "average_decisions": round(sum(row.get("step_count", 0) for row in grades) / len(grades), 2) if grades else 0,
    }


def run_suite(agent, cases: list[dict], output_dir: Path, label: str,
              rules: list[str] | None = None, use_active: bool = False,
              skill_version: int | None = None, memory_snapshot_id: str | None = None) -> dict:
    grades = []
    for case in cases:
        try:
            trace = agent.run_case(case, None if use_active else (rules or []), version=label,
                                   skill_version=skill_version, memory_snapshot_id=memory_snapshot_id)
            with ls_trace("evaluator", run_type="chain", inputs={"case_id": case["id"],
                          "trace_id": trace["trace_id"]}, metadata={"experiment": label}) as span:
                result = grade(case, trace)
                span.end(outputs=result)
            save_json(output_dir / "traces" / label / f"{case['id']}.json", trace)
        except Exception as exc:
            result = {"case_id": case["id"], "pass": False, "error": f"{type(exc).__name__}: {exc}"}
        grades.append(result)
        print(f"  {case['id']}: {'PASS' if result['pass'] else 'FAIL'}")
    report = {"label": label, "summary": summarize(grades), "grades": grades}
    save_json(output_dir / f"{label}.json", report)
    return report
