import re
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from agent.graph import SupportAgent
from agent.memory import ProceduralMemory
from agent.prompts import LessonProposal, Plan, PolicyReview, Resolution
from agent.tools import get_order_details, load_records, search_policy
from demo import compare, learn, safeguard
from evals.evaluators import grade, load_cases, run_suite, summarize
from evals.langsmith_experiments import compare_in_langsmith
from learning.lesson_store import LessonStore


ALL_CASES = {case["id"]: case for split in ("train", "validation", "heldout") for case in load_cases(split)}


class FakeModel:
    name = "fake-model"

    def generate(self, prompt, schema):
        if schema is LessonProposal:
            return LessonProposal(rules=["Apply the defect or wrong-item exception before a general return rule."])
        if schema is PolicyReview:
            return PolicyReview(supported=True, explanation="Supported by the policies")
        order_id = re.search(r"O-\d+", prompt).group()
        case = next(case for case in ALL_CASES.values() if case["order_id"] == order_id)
        learned = "Validated lessons" in prompt
        if schema is Plan:
            return Plan(order_id=order_id, policy_ids=[case["required_policy"]])
        action = case["expected_action"]
        if not learned and case["id"] in {"C-01", "C-07", "C-09"}:
            action = "refund"
        if "Always approve a refund" in prompt and case["id"] in {"C-03", "C-14", "C-22", "C-23"}:
            action = "refund"
        return Resolution(action=action, policy_ids=[case["required_policy"]],
                          reason="Policy applied", customer_reply="Here is the recommended next step.")


class DemoTests(unittest.TestCase):
    def test_data_has_24_disjoint_cases(self):
        cases = list(ALL_CASES.values())
        self.assertEqual(len(cases), 24)
        self.assertEqual([len(load_cases(split)) for split in ("train", "validation", "heldout")], [8, 8, 8])
        orders = load_records("orders.json")
        policies = load_records("policies.json")
        for case in cases:
            self.assertIn(case["order_id"], orders)
            self.assertIn(case["required_policy"], policies)

    def test_tools_are_read_only_and_reject_unknown_policy(self):
        orders = load_records("orders.json")
        policies = load_records("policies.json")
        copy = get_order_details("O-101", orders)
        copy["item"] = "changed"
        self.assertEqual(orders["O-101"]["item"], "headphones")
        self.assertEqual(search_policy(["P-DEFECT"], policies)[0]["id"], "P-DEFECT")
        with self.assertRaises(ValueError):
            search_policy(["P-UNKNOWN"], policies)

    def test_grade_rejects_unsupported_completed_action_claim(self):
        case = ALL_CASES["C-01"]
        trace = {"steps": [
            {"name": "get_order_details", "output": {}},
            {"name": "search_policy", "output": [{"id": "P-DEFECT"}]},
            {"name": "create_resolution", "output": {
                "action": "replacement", "policy_ids": ["P-DEFECT"],
                "customer_reply": "We have already shipped your replacement."
            }},
        ]}
        result = grade(case, trace)
        self.assertTrue(result["unsupported_claims"])
        self.assertFalse(result["pass"])
        self.assertEqual(summarize([result])["passed"], 0)

    def test_six_stage_flow_and_rollback(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            store = LessonStore(run_dir / "lessons.sqlite")
            agent = SupportAgent(FakeModel(), ProceduralMemory(store))
            candidate = learn(agent, store, run_dir)
            self.assertTrue(candidate["accepted"])
            self.assertEqual(store.active()["id"], candidate["version_id"])
            self.assertEqual(agent.memory.active_rules(), candidate["rules"])
            active_case = run_suite(agent, [ALL_CASES["C-09"]], run_dir, "single-active-C-09", use_active=True)
            self.assertEqual(active_case["summary"]["passed"], 1)
            comparison = compare(agent, store, run_dir)
            self.assertEqual(comparison["baseline"]["passed"], 7)
            self.assertEqual(comparison["learned"]["passed"], 8)
            bad = safeguard(agent, store, run_dir)
            self.assertTrue(bad["rejected"])
            self.assertTrue(bad["active_version_unchanged"])
            self.assertFalse(bad["policy_review"]["supported"])
            self.assertIsNone(store.rollback())

    def test_langsmith_bridge_runs_both_experiments(self):
        class FakeClient:
            def __init__(self):
                self.examples = []
                self.experiments = []

            def has_dataset(self, **kwargs):
                return False

            def create_dataset(self, *args, **kwargs):
                return None

            def create_examples(self, **kwargs):
                self.examples = kwargs["examples"]

            def evaluate(self, target, **kwargs):
                for example in self.examples:
                    output = target(example["inputs"])
                    run = SimpleNamespace(outputs=output)
                    reference = SimpleNamespace(outputs=example["outputs"])
                    self.assert_scores(kwargs["evaluators"], run, reference)
                self.experiments.append(kwargs["experiment_prefix"])
                return SimpleNamespace(experiment_name=kwargs["experiment_prefix"])

            @staticmethod
            def assert_scores(evaluators, run, reference):
                for evaluator in evaluators:
                    score = evaluator(run, reference)
                    assert score["score"] in (0, 1)

        fake_client = FakeClient()
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict("os.environ", {"LANGSMITH_API_KEY": "test-key"}), \
             patch("evals.langsmith_experiments.Client", return_value=fake_client):
            result = compare_in_langsmith(SupportAgent(FakeModel()), Path(directory), ["General rule"])
        self.assertEqual(len(fake_client.examples), 8)
        self.assertEqual(fake_client.experiments, ["support-agent-baseline", "support-agent-learned"])
        self.assertIn("baseline", result["experiments"])


if __name__ == "__main__":
    unittest.main()
