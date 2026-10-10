"""Run the six stages of the customer-support improvement demo."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from agent.graph import GeminiModel, SupportAgent
from agent.memory import ProceduralMemory
from evals.evaluators import load_cases, run_suite, save_json
from learning.lesson_store import LessonStore
from learning.promotion_gate import evaluate_gate
from learning.reflection import propose_lesson, review_against_policy


def find_case(case_id: str) -> dict:
    for split in ("train", "validation", "heldout"):
        for case in load_cases(split):
            if case["id"] == case_id:
                return case
    raise ValueError(f"Unknown case: {case_id}")


def baseline(agent: SupportAgent, run_dir: Path) -> dict:
    print("Stage 1–3: baseline agent, observed failure, and evaluation")
    return run_suite(agent, load_cases("train"), run_dir, "baseline-train")


def learn(agent: SupportAgent, store: LessonStore, run_dir: Path) -> dict:
    train_path = run_dir / "baseline-train.json"
    training = json.loads(train_path.read_text()) if train_path.exists() else baseline(agent, run_dir)
    failed_ids = {row["case_id"] for row in training["grades"] if not row["pass"]}
    if not failed_ids:
        result = {"accepted": False, "reason": "No training failure; no lesson proposed."}
        save_json(run_dir / "candidate.json", result)
        return result
    failed_examples = []
    for case_id in sorted(failed_ids):
        trace_path = run_dir / "traces" / "baseline-train" / f"{case_id}.json"
        if trace_path.exists():
            case = find_case(case_id)
            grade = next(row for row in training["grades"] if row["case_id"] == case_id)
            failed_examples.append({
                "request": case["message"],
                "order_facts": agent.orders[case["order_id"]],
                "trace": json.loads(trace_path.read_text()),
                "evaluator_feedback": grade,
            })
    if not failed_examples:
        result = {"accepted": False, "reason": "Failures had no usable traces."}
        save_json(run_dir / "candidate.json", result)
        return result

    print("Stage 4: reflect, check policy, and store a candidate lesson")
    rules = propose_lesson(agent.model, failed_examples, list(agent.policies.values()))
    version = store.create_candidate(rules, sorted(failed_ids))
    review = review_against_policy(agent.model, rules, list(agent.policies.values()))
    store.record_review(version["id"], review)
    print("Validation: baseline")
    base_report = run_suite(agent, load_cases("validation"), run_dir, "baseline-validation")
    print("Validation: candidate")
    candidate_report = run_suite(agent, load_cases("validation"), run_dir, "candidate-validation", rules)
    gate = evaluate_gate(base_report, candidate_report, review)
    store.record_validation(version["id"], gate)
    if gate["accepted"]:
        store.promote(version["id"])
        if agent.memory:
            agent.memory.refresh()
    else:
        store.reject(version["id"], gate["reason"])
    result = {"version_id": version["id"], "rules": rules, **gate}
    save_json(run_dir / "candidate.json", result)
    return result


def compare(agent: SupportAgent, store: LessonStore, run_dir: Path, publish: bool = False) -> dict:
    candidate_path = run_dir / "candidate.json"
    if not candidate_path.exists():
        learn(agent, store, run_dir)
    candidate = json.loads(candidate_path.read_text())
    print("Stage 5: baseline on untouched holdout cases")
    baseline_report = run_suite(agent, load_cases("heldout"), run_dir, "baseline-heldout")
    if candidate["accepted"]:
        print("Stage 5: promoted version on the same holdout cases")
        learned_report = run_suite(agent, load_cases("heldout"), run_dir, "learned-heldout", candidate["rules"])
    else:
        learned_report = None
    result = {
        "candidate_accepted": candidate["accepted"],
        "baseline": baseline_report["summary"],
        "learned": learned_report["summary"] if learned_report else None,
        "baseline_grades": baseline_report["grades"],
        "learned_grades": learned_report["grades"] if learned_report else None,
    }
    save_json(run_dir / "comparison.json", result)
    if publish and learned_report:
        from evals.langsmith_experiments import compare_in_langsmith

        result["langsmith"] = compare_in_langsmith(agent, run_dir, candidate["rules"])
        save_json(run_dir / "comparison.json", result)
    return result


def safeguard(agent: SupportAgent, store: LessonStore, run_dir: Path) -> dict:
    print("Stage 6: reject an overgeneralized refund rule")
    active_before = store.active()
    bad_rule = "Always approve a refund for defective products."
    version = store.create_candidate([bad_rule], ["intentional-demo-update"])
    review = review_against_policy(agent.model, [bad_rule], list(agent.policies.values()))
    store.record_review(version["id"], review)
    safety_cases = [find_case(case_id) for case_id in ("C-04", "C-03", "C-14", "C-22", "C-23")]
    print("Policy-compliance regression suite")
    bad_report = run_suite(agent, safety_cases, run_dir, "bad-update-safety", [bad_rule])
    store.reject(version["id"], review["explanation"] if not review["supported"] else "Safety candidate is never auto-promoted")
    active_after = store.active()
    result = {
        "bad_version_id": version["id"],
        "rule": bad_rule,
        "policy_review": review,
        "safety_scores": bad_report["summary"],
        "rejected": store.get(version["id"])["status"] == "rejected",
        "active_version_unchanged": (active_before or {}).get("id") == (active_after or {}).get("id"),
        "rollback_command": "uv run python demo.py rollback",
    }
    save_json(run_dir / "bad-update.json", result)
    return result


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Six-stage Gemini + LangGraph improvement demo")
    parser.add_argument("command", choices=["case", "baseline", "learn", "compare", "safeguard", "rollback", "demo"])
    parser.add_argument("--case-id", default="C-01")
    parser.add_argument("--version", choices=["baseline", "active"], default="baseline",
                        help="Version to use for the `case` command")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--publish-langsmith", action="store_true", help="Run fresh LangSmith holdout experiments")
    args = parser.parse_args()

    root = Path("runs")
    latest = root / "latest.txt"
    if args.command == "rollback":
        if not latest.exists() and not args.run_dir:
            raise SystemExit("No run to roll back. Pass --run-dir or run the demo first.")
        run_dir = args.run_dir or Path(latest.read_text().strip())
        current = LessonStore(run_dir / "lessons.sqlite").rollback()
        print(json.dumps({"active_version": current["id"] if current else None}, indent=2))
        return

    try:
        model = GeminiModel()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if args.run_dir:
        run_dir = args.run_dir
    elif args.command in {"baseline", "demo"} or (args.command == "case" and not latest.exists()):
        run_dir = root / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    elif latest.exists():
        run_dir = Path(latest.read_text().strip())
    else:
        raise SystemExit("Start with `uv run python demo.py demo` or pass --run-dir.")
    run_dir.mkdir(parents=True, exist_ok=True)
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(str(run_dir.resolve()) + "\n")
    store = LessonStore(run_dir / "lessons.sqlite")
    agent = SupportAgent(model, ProceduralMemory(store))
    print(f"Artifacts: {run_dir.resolve()}")

    if args.command == "case":
        case = find_case(args.case_id)
        active = store.active() if args.version == "active" else None
        if args.version == "active" and active is None:
            raise SystemExit("No active lesson version. Run `uv run python demo.py learn` first.")
        result = run_suite(agent, [case], run_dir, f"single-{args.version}-{case['id']}",
                           use_active=args.version == "active")
    elif args.command == "baseline":
        result = baseline(agent, run_dir)
    elif args.command == "learn":
        result = learn(agent, store, run_dir)
    elif args.command == "compare":
        result = compare(agent, store, run_dir, args.publish_langsmith)
    elif args.command == "safeguard":
        result = safeguard(agent, store, run_dir)
    else:
        baseline(agent, run_dir)
        learn(agent, store, run_dir)
        compare(agent, store, run_dir, args.publish_langsmith)
        result = safeguard(agent, store, run_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
