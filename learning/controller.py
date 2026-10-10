"""Evidence-led, single-job procedural improvement controller."""
from __future__ import annotations

import difflib
import json
from pathlib import Path

from langsmith.run_helpers import trace as ls_trace

from agent.skills import SKILL_FILE, patched_skill
from evals.evaluators import load_cases, run_suite, save_json
from learning.promotion_gate import evaluate_gate
from learning.reflection import FINAL_SALE_DEFECT_RULE, propose_lesson, review_against_policy


SAFETY_IDS = ("C-02", "C-03", "C-04", "C-15", "C-16")


def diagnose_case(case: dict, trajectory: dict, grade: dict) -> dict:
    steps = trajectory["steps"]
    tools = [s for s in steps if s.get("type") == "tool"]
    retrieved = {p["id"] for s in tools if s.get("name") == "search_policy"
                 for p in (s.get("output") or [])}
    decision = next((s for s in reversed(steps) if s.get("type") == "decision"
                     and s.get("output", {}).get("resolution")), None)
    citations = [s["id"] for s in tools if s.get("name") == "search_policy"]
    proposed_ids = set((decision or {}).get("output", {}).get("resolution", {}).get("policy_ids", []))
    if case.get("required_policy") == "P-DEFECT" and "final-sale" in case["message"].lower() and \
            ("P-FINAL" in retrieved or "P-FINAL" in proposed_ids) and \
            (decision or {}).get("output", {}).get("resolution", {}).get("action") != case["expected_action"] and \
            "final-sale item follows P-DEFECT" not in SKILL_FILE.read_text():
        category, target = "skill_instruction", "skill"
        evidence = ["s1", *citations, *([decision["id"]] if decision else [])]
    elif any(s.get("error") for s in tools) or case["required_policy"] not in retrieved:
        category, target = "tool_selection_execution", "skill"
        evidence = [s["id"] for s in tools if s.get("error") or s.get("name") == "search_policy"]
    elif trajectory.get("retrieved_memory_ids") and grade.get("stale_memory_reliance"):
        category, target = "memory_retrieval_use", "memory_rule"
        evidence = [s["id"] for s in tools if s.get("name") == "retrieve_case_memory"]
    else:
        category, target = "answer_synthesis", "skill"
        evidence = [decision["id"]] if decision else [s["id"] for s in steps[-2:]]
    return {"case_id": case["id"], "trace_id": trajectory["trace_id"], "category": category,
            "target": target, "evidence_step_ids": evidence, "policy_ids": sorted(retrieved),
            "observed_action": trajectory["resolution"]["action"],
            "expected_action": case["expected_action"], "grade": grade}


class ImprovementController:
    def __init__(self, agent, store, run_dir: Path):
        self.agent, self.store, self.run_dir = agent, store, Path(run_dir)

    def enqueue(self, case: dict, trajectory: dict, grade: dict) -> dict:
        if grade["pass"]:
            raise ValueError("Only externally assessed failures can trigger improvement")
        path = self.run_dir / "jobs" / f"{trajectory['trace_id']}.json"
        if path.exists():
            return json.loads(path.read_text())
        job = {"trace_id": trajectory["trace_id"], "case_id": case["id"],
               "source_version": trajectory["skill"], "grade": grade, "status": "queued"}
        save_json(path, job)
        return job

    def record_assessed_case(self, case: dict, trajectory: dict, grade: dict,
                             customer_id: str) -> str:
        """Persist a minimal proposed outcome only after external scoring."""
        if not self.agent.case_memory or grade.get("case_id") != case.get("id"):
            raise ValueError("An assessed matching case and case-memory store are required")
        order = self.agent.orders[case["order_id"]]
        issue_type = ("wrong_item" if order.get("wrong_item") else "defect" if order.get("defect_confirmed")
                      else "delivery" if order.get("days_since_delivery") is None else "return")
        summary = f"{issue_type}; delivery age={order.get('days_since_delivery')}; stock={order.get('replacement_stock')}"
        from agent.graph import policy_hash
        return self.agent.case_memory.write_case_memory(customer_id=customer_id, order_id=case["order_id"],
            issue_type=issue_type, issue_summary=summary,
            policy_ids=trajectory["resolution"].get("policy_ids", []),
            policy_hash=policy_hash(self.agent.policies),
            recommendation=trajectory["resolution"].get("action"),
            source_trace_id=trajectory["trace_id"], source_type="agent_recommendation",
            outcome_status="proposed")

    def diagnose(self, job: dict, case: dict, trajectory: dict) -> dict:
        with ls_trace("improvement.diagnose", run_type="chain", inputs={"trace_id": job["trace_id"]}) as span:
            diagnosis = diagnose_case(case, trajectory, job["grade"])
            span.end(outputs=diagnosis)
        job = {**job, "diagnosis": diagnosis, "status": "diagnosed"}
        save_json(self.run_dir / "jobs" / f"{job['trace_id']}.json", job)
        return job

    def propose(self, job: dict, case: dict, trajectory: dict) -> dict:
        if job.get("diagnosis", {}).get("target") not in {"skill", "memory_rule"}:
            raise ValueError("Controller can edit only the allowlisted skill or memory-use rule")
        old = self.store.active()["content"] if self.store.active() else SKILL_FILE.read_text()
        policies = list(self.agent.policies.values())
        if job["diagnosis"]["category"] == "memory_retrieval_use":
            rule = ("A prior recommendation on another order or under another policy version is contextual history, "
                    "not evidence of current eligibility. Proposed outcomes are not confirmed precedents. "
                    "Recheck the current order and current policy before recommending a remedy.")
        elif job["diagnosis"]["category"] == "skill_instruction" and case["required_policy"] == "P-DEFECT":
            rule = FINAL_SALE_DEFECT_RULE
        else:
            rules = propose_lesson(self.agent.model, [{"request": case["message"], "trace": trajectory,
                 "feedback": job["grade"]}], policies)
            rule = rules[0]
        new = patched_skill(old, rule)
        review = review_against_policy(self.agent.model, [rule], policies)
        candidate = self.store.create_candidate([rule], [case["id"]], content=new,
            diagnosis=job["diagnosis"], evidence=[job["trace_id"]])
        self.store.record_review(candidate["id"], review)
        diff = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                            fromfile="skills-old.md", tofile="skills-candidate.md"))
        result = {**job, "candidate_id": candidate["id"], "candidate_hash": candidate["content_hash"],
                  "rule": rule, "diff": diff, "policy_review": review, "status": "proposed"}
        save_json(self.run_dir / "jobs" / f"{job['trace_id']}.json", result)
        return result

    def validate(self, job: dict) -> dict:
        candidate = self.store.get(job["candidate_id"])
        validation = load_cases("validation")
        if self.agent.case_memory:
            from agent.graph import SupportAgent
            frozen, snapshot = self.agent.case_memory.snapshot(self.run_dir / "validation-memory.sqlite")
            eval_agent = SupportAgent(self.agent.model, case_memory=frozen, skill_store=self.store)
        else:
            eval_agent, snapshot = self.agent, "no-memory"
        with ls_trace("improvement.candidate_eval", run_type="chain",
                      inputs={"candidate_id": candidate["id"]}) as span:
            base = run_suite(eval_agent, validation, self.run_dir, "baseline-validation", rules=[],
                             memory_snapshot_id=snapshot)
            updated = run_suite(eval_agent, validation, self.run_dir, "candidate-validation",
                                skill_version=candidate["id"], use_active=True,
                                memory_snapshot_id=snapshot)
            safety_cases = [case for case in load_cases("train") + validation if case["id"] in SAFETY_IDS]
            safety = run_suite(eval_agent, safety_cases, self.run_dir, "candidate-safety",
                               skill_version=candidate["id"], use_active=True,
                               memory_snapshot_id=snapshot)
            gate = evaluate_gate(base, updated, candidate["policy_review"], safety)
            span.end(outputs=gate)
        self.store.record_validation(candidate["id"], gate)
        result = {**job, "gate": gate, "status": "validated"}
        save_json(self.run_dir / "jobs" / f"{job['trace_id']}.json", result)
        return result

    def refresh_review(self, job: dict) -> dict:
        candidate = self.store.get(job["candidate_id"])
        review = review_against_policy(self.agent.model, candidate["rules"],
                                       list(self.agent.policies.values()))
        self.store.record_review(candidate["id"], review)
        result = {**job, "policy_review": review}
        save_json(self.run_dir / "jobs" / f"{job['trace_id']}.json", result)
        return result

    def decide(self, job: dict) -> dict:
        with ls_trace("improvement.promotion", run_type="chain",
                      inputs={"candidate_id": job["candidate_id"]}) as span:
            if job["gate"]["accepted"]:
                version = self.store.promote(job["candidate_id"])
                status = "promoted"
            else:
                version = self.store.reject(job["candidate_id"], job["gate"]["reason"])
                status = "rejected"
            span.end(outputs={"status": status, "version_id": version["id"]})
        result = {**job, "status": status, "rollback_pointer": version.get("parent_id")}
        save_json(self.run_dir / "jobs" / f"{job['trace_id']}.json", result)
        return result
