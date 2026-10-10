"""Independent, inspectable scoring of support-agent results."""

from __future__ import annotations

import json
import re
from pathlib import Path


CASE_DIR = Path(__file__).parent
SPLITS = {
    "train": "train_cases.json",
    "validation": "validation_cases.json",
    "heldout": "heldout_cases.json",
}
REQUIRED_TOOLS = {"get_order_details", "search_policy", "create_resolution"}
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
    output = steps[-1]["output"]
    names = {step.get("name") for step in steps}
    retrieved = {
        policy["id"]
        for step in steps if step.get("name") == "search_policy"
        for policy in step["output"]
    }
    cited = set(output["policy_ids"])
    resolution_correct = output["action"] == case["expected_action"]
    required_tool_usage = REQUIRED_TOOLS <= names
    policy_compliance = case["required_policy"] in cited and bool(cited) and cited <= retrieved
    unsupported_claims = bool(COMPLETED_ACTION.search(output["customer_reply"]))
    return {
        "case_id": case["id"],
        "pass": resolution_correct and required_tool_usage and policy_compliance and not unsupported_claims,
        "resolution_correct": resolution_correct,
        "required_tool_usage": required_tool_usage,
        "policy_compliance": policy_compliance,
        "unsupported_claims": unsupported_claims,
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
    }


def run_suite(agent, cases: list[dict], output_dir: Path, label: str,
              rules: list[str] | None = None, use_active: bool = False) -> dict:
    grades = []
    for case in cases:
        try:
            trace = agent.run_case(case, None if use_active else (rules or []), version=label)
            result = grade(case, trace)
            save_json(output_dir / "traces" / label / f"{case['id']}.json", trace)
        except ValueError as exc:
            result = {"case_id": case["id"], "pass": False, "error": f"{type(exc).__name__}: {exc}"}
        grades.append(result)
        print(f"  {case['id']}: {'PASS' if result['pass'] else 'FAIL'}")
    report = {"label": label, "summary": summarize(grades), "grades": grades}
    save_json(output_dir / f"{label}.json", report)
    return report
