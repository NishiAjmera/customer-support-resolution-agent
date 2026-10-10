"""Turn grounded training failures into policy-checked candidate rules."""

from __future__ import annotations

import json
import re

from agent.prompts import LessonProposal, PolicyReview


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
    """Reject known contradictions, then request a source-grounded Gemini review."""
    for rule in rules:
        lowered = rule.lower()
        if re.search(r"\balways\b.*\brefund\b.*\bdefect", lowered) or re.search(
            r"\balways\b.*\bdefect\b.*\brefund", lowered
        ):
            return {
                "supported": False,
                "explanation": "Unconditional defect refunds contradict P-DEFECT's replacement and warranty windows.",
            }
    prompt = (
        "You are a strict policy reviewer. Decide whether every proposed rule follows from "
        "the supplied policies. Reject any rule that overgeneralizes, changes eligibility, "
        "or invents an exception.\n"
        f"Policies: {json.dumps(policies)}\nRules: {json.dumps(rules)}"
    )
    return model.generate(prompt, PolicyReview).model_dump()
