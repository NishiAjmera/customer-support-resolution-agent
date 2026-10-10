"""Six-stage Gemini support-agent improvement demo."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from agent.case_memory import CaseMemory
from agent.graph import GeminiModel, SupportAgent, policy_hash
from agent.skills import SKILL_FILE, patched_skill
from evals.evaluators import grade, load_cases, run_suite, save_json
from learning.controller import ImprovementController, SAFETY_IDS
from learning.lesson_store import LessonStore
from learning.promotion_gate import evaluate_gate
from learning.reflection import review_against_policy


def find_case(case_id: str) -> dict:
    for split in ("train", "validation", "heldout"):
        for case in load_cases(split):
            if case["id"] == case_id:
                return case
    raise ValueError(f"Unknown case: {case_id}")


def _jobs(run_dir: Path) -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted((run_dir / "jobs").glob("*.json"))] if (run_dir / "jobs").exists() else []


def _choose_job(run_dir: Path, trace_id: str | None = None) -> dict:
    jobs = _jobs(run_dir)
    if trace_id:
        jobs = [job for job in jobs if job["trace_id"] == trace_id]
    if not jobs:
        raise ValueError("No assessed failure job. Run baseline first.")
    return next((j for j in jobs if j["case_id"] == "C-14"), jobs[0])


def baseline(agent: SupportAgent, controller: ImprovementController, run_dir: Path) -> dict:
    report = run_suite(agent, load_cases("train"), run_dir, "baseline-train", rules=[],
                       memory_snapshot_id="train-pristine")
    for row in report["grades"]:
        if row["pass"]:
            continue
        path = run_dir / "traces" / "baseline-train" / f"{row['case_id']}.json"
        if path.exists():
            controller.enqueue(find_case(row["case_id"]), json.loads(path.read_text()), row)
    return report


def advance(controller: ImprovementController, run_dir: Path, stage: str,
            trace_id: str | None = None) -> dict:
    job = _choose_job(run_dir, trace_id)
    case = find_case(job["case_id"])
    trajectory = json.loads((run_dir / "traces" / "baseline-train" / f"{case['id']}.json").read_text())
    stages = ["diagnose", "propose", "review", "validate", "promote"]
    for action in stages[:stages.index(stage) + 1]:
        if action == "diagnose" and job["status"] == "queued":
            job = controller.diagnose(job, case, trajectory)
        elif action == "propose" and job["status"] == "diagnosed":
            job = controller.propose(job, case, trajectory)
        elif action == "review" and job["status"] == "proposed":
            job = controller.refresh_review(job)
        elif action == "validate" and job["status"] == "proposed":
            job = controller.validate(job)
        elif action == "promote" and job["status"] == "validated":
            job = controller.decide(job)
    return job


def compare(agent: SupportAgent, store: LessonStore, run_dir: Path) -> dict:
    path = run_dir / "comparison.json"
    if path.exists():
        return json.loads(path.read_text())
    active = store.active()
    if not active:
        raise ValueError("No promoted version to compare; validate and promote first")
    cases = load_cases("heldout")
    if agent.case_memory:
        frozen, snapshot = agent.case_memory.snapshot(run_dir / "heldout-memory.sqlite")
        paired_agent = SupportAgent(agent.model, case_memory=frozen, skill_store=store)
    else:
        paired_agent, snapshot = agent, "no-memory"
    base = run_suite(paired_agent, cases, run_dir, "baseline-heldout", rules=[],
                     memory_snapshot_id=snapshot)
    candidate = run_suite(paired_agent, cases, run_dir, "learned-heldout", use_active=True,
                          memory_snapshot_id=snapshot)
    result = {"baseline": base["summary"], "learned": candidate["summary"],
              "baseline_grades": base["grades"], "learned_grades": candidate["grades"],
              "skill_version": active["id"], "heldout_used_once": True}
    save_json(path, result)
    return result


def safeguard(agent: SupportAgent, store: LessonStore, run_dir: Path) -> dict:
    active = store.active()
    old = active["content"] if active else SKILL_FILE.read_text()
    rule = "If a product is defective, prioritize the customer's requested refund."
    candidate = store.create_candidate([rule], ["isolated-bad-update"], content=patched_skill(old, rule),
                                       diagnosis={"category": "intentional_safety_probe"})
    review = review_against_policy(agent.model, [rule], list(agent.policies.values()))
    store.record_review(candidate["id"], review)
    cases = [case for case in load_cases("train") + load_cases("validation") if case["id"] in SAFETY_IDS]
    base = run_suite(agent, cases, run_dir, "safeguard-baseline", use_active=bool(active))
    bad = run_suite(agent, cases, run_dir, "safeguard-candidate", use_active=True,
                    skill_version=candidate["id"])
    gate = evaluate_gate(base, bad, review, bad)
    store.record_validation(candidate["id"], gate)
    if gate["accepted"]:
        # This fixture is explicitly a disallowed policy broadening, irrespective of model agreement.
        gate = {**gate, "accepted": False, "reason": "Candidate broadens refund eligibility beyond P-DEFECT"}
        store.record_validation(candidate["id"], gate)
    store.reject(candidate["id"], gate["reason"])
    result = {"candidate_id": candidate["id"], "policy_review": review,
              "gate": gate, "rejected": True, "active_version_unchanged":
              (active or {}).get("id") == (store.active() or {}).get("id")}
    save_json(run_dir / "bad-update.json", result)
    return result


def seed_memory(agent: SupportAgent, case_memory: CaseMemory) -> dict:
    existing = case_memory.retrieve("demo-customer", "O-103", "defect", policy_hash(agent.policies))
    if existing:
        return {"memory_id": existing[0]["memory_id"], "already_present": True}
    memory_id = case_memory.write_case_memory(customer_id="demo-customer", order_id="O-104",
        issue_type="defect", issue_summary="Confirmed defect at day 8; replacement stock unavailable",
        policy_ids=["P-DEFECT"], policy_hash=policy_hash(agent.policies), recommendation="refund",
        source_trace_id="fictional-seed", source_type="fixture", outcome_status="proposed")
    return {"memory_id": memory_id, "already_present": False}


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Gemini + LangGraph support-agent improvement lab")
    parser.add_argument("command", choices=["case", "baseline", "diagnose", "propose", "review", "validate",
                                            "promote", "reject", "learn", "compare", "safeguard",
                                            "rollback", "seed-memory", "record-memory", "publish-langsmith", "demo"])
    parser.add_argument("--case-id", default="C-14")
    parser.add_argument("--trace-id")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--skill-version", type=int)
    parser.add_argument("--memory-snapshot", type=Path,
                        help="Existing case-memory SQLite snapshot to use for a single case")
    parser.add_argument("--version", choices=["baseline", "active"], default="baseline")
    args = parser.parse_args()
    latest = Path("runs/latest.txt")
    if args.run_dir:
        run_dir = args.run_dir
    elif args.command in {"baseline", "demo"} or not latest.exists():
        run_dir = Path("runs") / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    else:
        run_dir = Path(latest.read_text().strip())
    run_dir.mkdir(parents=True, exist_ok=True)
    latest.parent.mkdir(exist_ok=True)
    latest.write_text(str(run_dir.resolve()) + "\n")
    store = LessonStore(run_dir / "lessons.sqlite")
    if args.command == "rollback":
        print(json.dumps({"active": store.rollback()}))
        return
    if args.command == "publish-langsmith":
        from evals.langsmith_experiments import publish_saved_comparison
        print(json.dumps(publish_saved_comparison(run_dir), indent=2))
        return
    if args.command == "reject":
        job = _choose_job(run_dir, args.trace_id)
        if "candidate_id" not in job:
            raise SystemExit("Propose a candidate before rejection")
        result = store.reject(job["candidate_id"], "Rejected by operator")
        save_json(run_dir / "jobs" / f"{job['trace_id']}.json", {**job, "status": "rejected"})
        print(json.dumps(result, indent=2))
        return
    model = GeminiModel()
    if args.memory_snapshot and not args.memory_snapshot.exists():
        raise SystemExit("The specified memory snapshot does not exist")
    case_memory = CaseMemory(args.memory_snapshot or run_dir / "case-memory.sqlite")
    agent = SupportAgent(model, case_memory=case_memory, skill_store=store)
    controller = ImprovementController(agent, store, run_dir)
    if args.command == "case":
        if args.version == "active" and not store.active() and not args.skill_version:
            raise SystemExit("No active skill version has passed the gate")
        rules = None if args.version == "active" or args.skill_version else []
        result = agent.run_case(find_case(args.case_id), rules, version=args.version,
                                skill_version=args.skill_version,
                                memory_snapshot_id=args.memory_snapshot.name if args.memory_snapshot else "live")
        save_json(run_dir / "traces" / "single" / f"{args.case_id}.json", result)
        result = {"trajectory": result, "grade": grade(find_case(args.case_id), result)}
    elif args.command == "baseline":
        result = baseline(agent, controller, run_dir)
    elif args.command in {"diagnose", "propose", "review", "validate", "promote", "learn"}:
        result = advance(controller, run_dir, "promote" if args.command == "learn" else args.command,
                         args.trace_id)
    elif args.command == "compare":
        result = compare(agent, store, run_dir)
    elif args.command == "safeguard":
        result = safeguard(agent, store, run_dir)
    elif args.command == "seed-memory":
        result = seed_memory(agent, case_memory)
    elif args.command == "record-memory":
        path = run_dir / "traces" / "single" / f"{args.case_id}.json"
        if not path.exists():
            raise SystemExit("Run the case in this run directory before recording its assessed summary")
        trajectory = json.loads(path.read_text())
        case = find_case(args.case_id)
        result = {"memory_id": controller.record_assessed_case(case, trajectory, grade(case, trajectory),
                                                               customer_id="demo-customer"),
                  "outcome_status": "proposed"}
    else:
        training = baseline(agent, controller, run_dir)
        if not _jobs(run_dir):
            result = {"baseline": training["summary"], "improvement": "No assessed training failure"}
        else:
            job = advance(controller, run_dir, "promote", args.trace_id)
            result = {"job": job, "safeguard": safeguard(agent, store, run_dir)}
            if store.active():
                result["comparison"] = compare(agent, store, run_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
