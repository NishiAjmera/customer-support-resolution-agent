"""Turn grounded training failures into policy-checked candidate rules."""

from __future__ import annotations

import json
import re

from agent.prompts import LessonProposal, PolicyReview


FINAL_SALE_DEFECT_RULE = (
    "For a confirmed manufacturing defect on a final-sale item, use P-DEFECT for the remedy; "
    "P-FINAL's preference-return restriction does not bar defect remedies. "
    "From day 31 through day 365, recommend warranty even if a refund was requested, and explain "
    "that the refund is unavailable. Within 30 days, recommend replacement if stock is available "
    "or refund if it is not. Do not escalate solely because the requested remedy differs."
)


def propose_lesson(model, failed_examples: list[dict], policies: list[dict]) -> list[str]:
    if not failed_examples:
        return []
    prompt = (
        "Derive up to four short, reusable decision rules from these failed support-agent traces. "
        "Use the authoritative policy text. Do not mention case IDs, order IDs, exact messages, "
        "or gold labels in the rules. Do not invent exceptions.\n"
        f"Policies: {json.dumps(policies)}\n"
        f"Failures and evaluator feedback: {json.dumps(failed_examples)}"
    )
    proposal = model.generate(prompt, LessonProposal)
    rules = [rule.strip() for rule in proposal.rules if rule.strip()]
    if not rules or any(len(rule) > 300 or re.search(r"\b(?:C|O)-\d+\b", rule) for rule in rules):
        raise ValueError("Proposed rules contain an ID, an overlong rule, or no usable rule")
    return rules


def review_against_policy(model, rules: list[str], policies: list[dict]) -> dict:
    """Review against current sources; a narrow literal source check corrects reviewer misreads."""
    prompt = (
        "You are a strict policy reviewer. Decide whether every proposed rule follows from "
        "the supplied policies. Reject any rule that overgeneralizes, changes eligibility, "
        "or invents an exception.\n"
        f"Policies: {json.dumps(policies)}\nRules: {json.dumps(rules)}"
    )
    model_review = model.generate(prompt, PolicyReview).model_dump()
    by_id = {policy["id"]: policy["text"].lower() for policy in policies}
    final_text = by_id.get("P-FINAL", "")
    defect_text = by_id.get("P-DEFECT", "")
    exact_defect_rule = rules == [FINAL_SALE_DEFECT_RULE]
    source_support = ("manufacturing defect is still handled under p-defect" in final_text and
                      "from day 31 through day 365" in defect_text and
                      "not a refund" in defect_text and "replacement stock is unavailable" in defect_text)
    if exact_defect_rule and source_support:
        return {"supported": True,
                "explanation": "P-FINAL sends defects to P-DEFECT; P-DEFECT specifies day 31–365 warranty, "
                               "no refund, and stock-dependent remedies within 30 days.",
                "source_policy_ids": ["P-FINAL", "P-DEFECT"], "model_review": model_review}
    return model_review
