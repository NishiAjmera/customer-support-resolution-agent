"""Validation and safeguard gates for lesson promotion."""


def evaluate_gate(baseline: dict, candidate: dict, policy_review: dict,
                  safety: dict | None = None) -> dict:
    base_rows = {row["case_id"]: row for row in baseline["grades"]}
    new_rows = {row["case_id"]: row for row in candidate["grades"]}
    if set(base_rows) != set(new_rows):
        raise ValueError("Baseline and candidate must use identical validation cases")
    fixed = sorted(case_id for case_id in base_rows if not base_rows[case_id]["pass"] and new_rows[case_id]["pass"])
    regressed = sorted(case_id for case_id in base_rows if base_rows[case_id]["pass"] and not new_rows[case_id]["pass"])
    new_critical = sorted(case_id for case_id in base_rows if new_rows[case_id].get("critical_violation")
                          and not base_rows[case_id].get("critical_violation"))
    safety_passed = safety is None or safety["summary"]["passed"] == safety["summary"]["total"]
    accepted = bool(policy_review["supported"] and fixed and not regressed and not new_critical and safety_passed)
    return {
        "accepted": accepted,
        "policy_review": policy_review,
        "fixed": fixed,
        "regressed": regressed,
        "new_critical": new_critical,
        "safety_passed": safety_passed,
        "baseline": baseline["summary"],
        "candidate": candidate["summary"],
        "reason": (
            "Improved validation without regression and passed policy review" if accepted
            else "Rejected: policy review failed, no validation improvement, or a regression occurred"
        ),
    }
