"""Deterministic orchestration tests; these are not live Gemini results."""
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.case_memory import CaseMemory
from agent.conversation import ConversationService
from agent.graph import Decision, SupportAgent, expected_from_facts, policy_hash
from agent.prompts import LessonProposal, PolicyReview, Resolution
from agent.tools import load_records
from demo import advance, baseline, compare, safeguard
from evals.evaluators import grade, load_cases
from evals.langsmith_experiments import publish_saved_comparison
from learning.controller import ImprovementController, diagnose_case
from learning.lesson_store import LessonStore

CASES = {case["id"]: case for split in ("train", "validation", "heldout") for case in load_cases(split)}
ORDERS = load_records("orders.json")


class FakeModel:
    name = "deterministic-test-model"

    def generate(self, prompt, schema):
        if schema is PolicyReview:
            supported = "prioritize the customer's requested refund" not in prompt.lower()
            return PolicyReview(supported=supported, explanation="Checked against fixture policy text")
        if schema is LessonProposal:
            return LessonProposal(rules=["Apply the applicable exception before general return rules."])
        order_id = re.search(r"Intake order ID: (O-\d{3})", prompt).group(1)
        order = ORDERS[order_id]
        action, policy = expected_from_facts(order)
        # The baseline's plausible skill gap is deliberately simulated for orchestration tests.
        if order_id in {"O-107", "O-114"} and "For a confirmed manufacturing defect on a final-sale item" not in prompt:
            policy, action = "P-FINAL", "deny"
        if "prioritize the customer's requested refund" in prompt.lower() and order.get("defect_confirmed"):
            action = "refund"
        if "Current order: null" in prompt:
            return Decision(action="tool", tool_name="get_order_details", order_id=order_id)
        if "Retrieved policies: []" in prompt:
            return Decision(action="tool", tool_name="search_policy", policy_ids=[policy])
        return Decision(action="finish", resolution=Resolution(action=action, policy_ids=[policy],
            reason="Fixture policy applied", customer_reply="I recommend the next policy-supported step."))


class SequenceModel:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.name = "sequence-test-model"

    def generate(self, prompt, schema):
        return next(self.decisions)


class AgentTests(unittest.TestCase):
    def test_data_and_policy_contract(self):
        self.assertEqual([len(load_cases(s)) for s in ("train", "validation", "heldout")], [8, 8, 8])
        for case in CASES.values():
            action, policy = expected_from_facts(ORDERS[case["order_id"]])
            self.assertEqual((action, policy), (case["expected_action"], case["required_policy"]))

    def test_trajectory_and_policy_safety(self):
        agent = SupportAgent(FakeModel())
        good = agent.run_case(CASES["C-03"], rules=[])
        self.assertEqual([s["name"] for s in good["steps"] if s["type"] == "tool"],
                         ["get_order_details", "search_policy"])
        self.assertTrue(grade(CASES["C-03"], good)["pass"])
        bad = agent.run_case(CASES["C-14"], rules=[])
        self.assertEqual(bad["resolution"]["action"], "escalate")
        self.assertFalse(grade(CASES["C-14"], bad)["pass"])
        self.assertTrue(any(s.get("safety_reason") for s in bad["steps"]))

    def test_unsupported_repeat_and_exhaustion_stop(self):
        unknown = SupportAgent(SequenceModel([Decision(action="tool", tool_name="refund_now")]))
        self.assertEqual(unknown.run_case(CASES["C-01"], rules=[])["status"], "escalated")
        bad_args = SupportAgent(SequenceModel([Decision(action="tool", tool_name="get_order_details",
            order_id="bad"), Decision(action="tool", tool_name="get_order_details", order_id="bad")]))
        trace = bad_args.run_case(CASES["C-01"], rules=[])
        self.assertEqual(trace["status"], "escalated")
        self.assertIn("Repeated", trace["resolution"]["reason"])
        many = [Decision(action="tool", tool_name="search_policy", policy_ids=["P-DEFECT"])]
        many += [Decision(action="tool", tool_name="get_order_details", order_id=f"O-{100+i:03d}") for i in range(1, 10)]
        exhausted = SupportAgent(SequenceModel(many)).run_case(CASES["C-01"], rules=[])
        self.assertEqual(exhausted["status"], "escalated")

    def test_memory_scope_and_unverified_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = CaseMemory(Path(directory) / "memory.sqlite")
            digest = policy_hash(load_records("policies.json"))
            mid = memory.write_case_memory(customer_id="demo-a", order_id="O-104", issue_type="defect",
                issue_summary="Confirmed day-eight defect, no stock", policy_ids=["P-DEFECT"],
                policy_hash=digest, recommendation="refund", source_trace_id="seed", source_type="fixture")
            later = memory.retrieve("demo-a", "O-103", "speaker failed with defect", digest)
            self.assertEqual(later[0]["memory_id"], mid)
            self.assertTrue(later[0]["stale"])
            self.assertEqual(later[0]["trust_level"], "context_only")
            self.assertEqual(memory.retrieve("demo-b", "O-103", "defect", digest), [])
            frozen, snapshot_id = memory.snapshot(Path(directory) / "frozen.sqlite")
            self.assertEqual(len(snapshot_id), 16)
            with self.assertRaises(ValueError):
                memory.write_case_memory(customer_id="demo-a", order_id="O-103", issue_type="defect",
                    issue_summary="No confirmation", policy_ids=[], policy_hash=digest,
                    recommendation="refund", source_trace_id="bad", outcome_status="human_confirmed")
            memory.tombstone(mid, "Fixture corrected", "correction-trace")
            self.assertEqual(memory.retrieve("demo-a", "O-103", "defect", digest), [])
            self.assertEqual(frozen.retrieve("demo-a", "O-103", "defect", digest)[0]["memory_id"], mid)

    def test_observed_optional_memory_and_policy_first_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = CaseMemory(Path(directory) / "memory.sqlite")
            mid = memory.write_case_memory(customer_id="demo-a", order_id="O-104", issue_type="defect",
                issue_summary="Confirmed day-eight defect, no stock", policy_ids=["P-DEFECT"],
                policy_hash=policy_hash(load_records("policies.json")), recommendation="refund",
                source_trace_id="prior-assessed-case", source_type="fixture")
            model = SequenceModel([
                Decision(action="tool", tool_name="retrieve_case_memory"),
                Decision(action="tool", tool_name="search_policy", policy_ids=["P-DEFECT"]),
                Decision(action="tool", tool_name="get_order_details", order_id="O-103"),
                Decision(action="finish", resolution=Resolution(action="warranty", policy_ids=["P-DEFECT"],
                    reason="Day 75 follows warranty", customer_reply="I recommend a warranty review.")),
            ])
            agent = SupportAgent(model, case_memory=memory)
            case = {**CASES["C-03"], "customer_id": "demo-a"}
            trace = agent.run_case(case, rules=[])
            self.assertEqual([s["name"] for s in trace["steps"] if s["type"] == "tool"],
                             ["retrieve_case_memory", "search_policy", "get_order_details"])
            self.assertEqual(trace["retrieved_memory_ids"], [mid])
            self.assertEqual(trace["resolution"]["action"], "warranty")
            self.assertTrue(grade(case, trace)["pass"])
            bad_model = SequenceModel([
                Decision(action="tool", tool_name="retrieve_case_memory"),
                Decision(action="tool", tool_name="get_order_details", order_id="O-103"),
                Decision(action="tool", tool_name="search_policy", policy_ids=["P-DEFECT"]),
                Decision(action="finish", resolution=Resolution(action="refund", policy_ids=["P-DEFECT"],
                    reason="Copied prior outcome", customer_reply="I recommend a refund.")),
            ])
            bad_trace = SupportAgent(bad_model, case_memory=memory).run_case(case, rules=[])
            bad_grade = grade(case, bad_trace)
            self.assertTrue(bad_trace["stale_memory_reliance"])
            self.assertEqual(diagnose_case(case, bad_trace, bad_grade)["category"], "memory_retrieval_use")

    def test_conversation_missing_id_isolation_idempotency(self):
        service = ConversationService(SupportAgent(FakeModel()))
        first = service.handle_message("s1", "demo-a", "My speaker failed", turn_id="t1")
        self.assertEqual(first.status, "needs_order_id")
        self.assertEqual(service.handle_message("s1", "demo-a", "O-103", turn_id="t2").status, "answered")
        self.assertIn("My speaker failed", service.sessions["s1"]["issue"])
        self.assertEqual(service.handle_message("s1", "demo-a", "O-103", turn_id="t2").trace_id,
                         service.sessions["s1"]["last_trace_id"])
        self.assertEqual(service.handle_message("s2", "demo-b", "O-999", turn_id="t1").status, "needs_order_id")
        self.assertEqual(service.handle_message("s1", "demo-a", "O-ABC", turn_id="t3").status, "needs_order_id")
        self.assertEqual(service.sessions["s2"]["known_order_id"], None)
        self.assertEqual(service.handle_message("s1", "demo-a", "O-104", turn_id="t4").status,
                         "needs_issue")
        self.assertEqual(service.sessions["s1"]["known_order_id"], "O-104")
        with self.assertRaises(ValueError):
            service.handle_message("s1", "demo-b", "hello")

    def test_improvement_gate_persistence_and_holdout_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = LessonStore(root / "lessons.sqlite")
            agent = SupportAgent(FakeModel(), skill_store=store)
            controller = ImprovementController(agent, store, root)
            report = baseline(agent, controller, root)
            self.assertGreater(report["summary"]["passed"], 0)
            job = advance(controller, root, "promote")
            self.assertEqual(job["status"], "promoted")
            self.assertEqual(LessonStore(root / "lessons.sqlite").active()["id"], job["candidate_id"])
            self.assertIn("P-DEFECT", store.active()["content"])
            comparison = compare(agent, store, root)
            self.assertEqual(comparison, compare(agent, store, root))
            class FakeClient:
                def has_dataset(self, **kwargs):
                    return False
                def create_dataset(self, *args, **kwargs):
                    return None
                def create_examples(self, **kwargs):
                    self.examples = kwargs["examples"]
                def evaluate(self, target, **kwargs):
                    for example in self.examples:
                        output = target(example["inputs"])
                        self_case = SimpleNamespace(outputs=output)
                        reference = SimpleNamespace(outputs=example["outputs"])
                        for evaluator in kwargs["evaluators"]:
                            self.assert_score(evaluator(self_case, reference))
                    return SimpleNamespace(experiment_name=kwargs["experiment_prefix"])
                @staticmethod
                def assert_score(score):
                    assert score["score"] in (0, 1)
            with patch.dict("os.environ", {"LANGSMITH_API_KEY": "fake-key"}), \
                 patch("evals.langsmith_experiments.Client", return_value=FakeClient()):
                published = publish_saved_comparison(root)
            self.assertEqual(len(published["experiments"]), 2)
            safety = safeguard(agent, store, root)
            self.assertTrue(safety["rejected"])
            self.assertTrue(safety["active_version_unchanged"])
            store.rollback()
            self.assertIsNone(LessonStore(root / "lessons.sqlite").active())


if __name__ == "__main__":
    unittest.main()
